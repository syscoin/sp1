use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

use anyhow::{anyhow, Context, Result};

/// PVOR v1: minimal u32 table container.
///
/// Layout:
/// - MAGIC "PVOR" (4)
/// - VERSION u32 (1)
/// - num_tables u32
/// - reserved u32
/// - table metas:
///   - name_len u16
///   - name bytes
///   - rows u32
///   - cols u32
///   - elem_size u8 (must be 4)
///   - encoding u8 (0 = raw u32 LE)
///   - reserved u16
///   - offset u64 (from file start)
///   - byte_len u64
/// - table data blobs at offsets, row-major u32 little-endian
const MAGIC: [u8; 4] = *b"PVOR";
const VERSION: u32 = 1;

pub struct TableMeta<'a> {
    pub name: &'a str,
    pub rows: u32,
    pub cols: u32,
    pub values_u32_le: Box<dyn Iterator<Item = u32> + 'a>,
}

pub fn write_u32_tables_streaming(out: &mut File, tables: &mut [TableMeta<'_>]) -> Result<()> {
    let mut header_len: u64 = 4 + 4 + 4 + 4;
    for t in tables.iter() {
        header_len += 2 + (t.name.as_bytes().len() as u64) + 4 + 4 + 1 + 1 + 2 + 8 + 8;
    }

    let mut offsets = Vec::with_capacity(tables.len());
    let mut cur = header_len;
    for t in tables.iter() {
        let byte_len = (t.rows as u64)
            .checked_mul(t.cols as u64)
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| anyhow!("byte_len overflow"))?;
        offsets.push((cur, byte_len));
        cur = cur.checked_add(byte_len).ok_or_else(|| anyhow!("offset overflow"))?;
    }

    out.write_all(&MAGIC)?;
    out.write_all(&VERSION.to_le_bytes())?;
    out.write_all(&(tables.len() as u32).to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())?;

    for (i, t) in tables.iter().enumerate() {
        let name_bytes = t.name.as_bytes();
        let name_len: u16 = name_bytes
            .len()
            .try_into()
            .map_err(|_| anyhow!("name too long"))?;
        out.write_all(&name_len.to_le_bytes())?;
        out.write_all(name_bytes)?;
        out.write_all(&t.rows.to_le_bytes())?;
        out.write_all(&t.cols.to_le_bytes())?;
        out.write_all(&[4u8])?; // elem_size
        out.write_all(&[0u8])?; // encoding
        out.write_all(&0u16.to_le_bytes())?;
        let (off, byte_len) = offsets[i];
        out.write_all(&off.to_le_bytes())?;
        out.write_all(&byte_len.to_le_bytes())?;
    }

    for t in tables.iter_mut() {
        let expected = (t.rows as u64) * (t.cols as u64);
        for _ in 0..expected {
            let v = t
                .values_u32_le
                .next()
                .ok_or_else(|| anyhow!("table {} ended early while streaming values", t.name))?;
            out.write_all(&v.to_le_bytes())?;
        }
        if t.values_u32_le.next().is_some() {
            return Err(anyhow!("table {} iterator produced too many elements", t.name));
        }
    }

    Ok(())
}

#[derive(Clone, Debug)]
pub struct TableInfo {
    pub name: String,
    pub rows: u32,
    pub cols: u32,
    pub offset: u64,
    pub byte_len: u64,
}

pub struct PvorReader {
    f: File,
    tables: Vec<TableInfo>,
    by_name: BTreeMap<String, usize>,
}

impl PvorReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;

        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        anyhow::ensure!(magic == MAGIC, "bad PVOR magic");

        let version = read_u32_le(&mut f)?;
        anyhow::ensure!(version == VERSION, "unsupported PVOR version {version}");

        let num_tables = read_u32_le(&mut f)? as usize;
        let _reserved = read_u32_le(&mut f)?;

        let mut tables = Vec::with_capacity(num_tables);
        let mut by_name = BTreeMap::new();
        for i in 0..num_tables {
            let name_len = read_u16_le(&mut f)? as usize;
            let mut name_bytes = vec![0u8; name_len];
            f.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes).context("PVOR table name utf8")?;

            let rows = read_u32_le(&mut f)?;
            let cols = read_u32_le(&mut f)?;
            let mut elem_size = [0u8; 1];
            let mut encoding = [0u8; 1];
            f.read_exact(&mut elem_size)?;
            f.read_exact(&mut encoding)?;
            let _rsv = read_u16_le(&mut f)?;
            anyhow::ensure!(elem_size[0] == 4, "PVOR elem_size must be 4");
            anyhow::ensure!(encoding[0] == 0, "PVOR encoding must be 0 (raw u32)");

            let offset = read_u64_le(&mut f)?;
            let byte_len = read_u64_le(&mut f)?;
            let expected = (rows as u64)
                .checked_mul(cols as u64)
                .and_then(|x| x.checked_mul(4))
                .ok_or_else(|| anyhow!("table byte_len overflow"))?;
            anyhow::ensure!(
                byte_len == expected,
                "PVOR table {name} byte_len mismatch: got {byte_len}, expected {expected}"
            );

            by_name.insert(name.clone(), i);
            tables.push(TableInfo {
                name,
                rows,
                cols,
                offset,
                byte_len,
            });
        }

        Ok(Self { f, tables, by_name })
    }

    pub fn table(&self, name: &str) -> Option<&TableInfo> {
        self.by_name.get(name).map(|&i| &self.tables[i])
    }

    pub fn table_names(&self) -> impl Iterator<Item = &str> {
        self.tables.iter().map(|t| t.name.as_str())
    }

    pub fn read_row_u32(&mut self, table: &str, row: u32) -> Result<Vec<u32>> {
        let (rows, cols, offset) = {
            let t = self
                .table(table)
                .ok_or_else(|| anyhow!("missing table '{table}'"))?;
            (t.rows, t.cols, t.offset)
        };
        anyhow::ensure!(row < rows, "row out of range for {table}");
        let row_off = offset + (row as u64) * (cols as u64) * 4;
        self.f.seek(SeekFrom::Start(row_off))?;
        let mut bytes = vec![0u8; (cols as usize) * 4];
        self.f.read_exact(&mut bytes)?;
        let mut out = vec![0u32; cols as usize];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            out[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(out)
    }
}

fn read_u16_le(r: &mut impl Read) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32_le(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64_le(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}


