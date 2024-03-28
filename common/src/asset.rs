use std::{
    fs,
    io::{BufReader, Write},
    path::{self, PathBuf},
};

use anyhow::Context;
use bincode::{config, de, enc};

pub trait Asset: Sized + enc::Encode + de::Decode {
    fn save<P: AsRef<path::Path>>(&self, path: P) -> anyhow::Result<()> {
        let config = config::standard();

        let data = bincode::encode_to_vec(self, config)?;

        let mut file = fs::File::create(path)?;
        file.write_all(&data[..])?;

        Ok(())
    }
    fn load<P: AsRef<path::Path>>(path: P) -> anyhow::Result<Self> {
        let config = config::standard();

        let file = fs::File::open(&path)?;
        let mut buf = BufReader::new(file);

        Ok(bincode::decode_from_reader(&mut buf, config)?)
    }

    fn load_from_cargo_manifest_dir(name: &'static str) -> anyhow::Result<Self> {
        let config = config::standard();

        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("..");
        path.push("assets");
        path.push(name);

        let file = fs::File::open(&path).with_context(|| format!("{:?}", path.canonicalize()))?;
        let mut buf = BufReader::new(file);

        Ok(bincode::decode_from_reader(&mut buf, config)?)
    }
}
