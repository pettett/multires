use std::{
    fs,
    io::{BufReader, Write},
    path::{self, PathBuf},
};

use anyhow::Context;
use bincode::{config, de, enc};

pub trait Asset: Sized + enc::Encode + de::Decode {
    fn save(&self) -> anyhow::Result<()> {
        let config = config::standard();

        let data = bincode::encode_to_vec(self, config)?;

        let mut file = fs::File::create("asset.bin")?;
        file.write_all(&data[..])?;

        Ok(())
    }
    fn load() -> anyhow::Result<Self> {
        let config = config::standard();

        let path = "asset.bin";

        let file = fs::File::open(&path)?;
        let mut buf = BufReader::new(file);

        Ok(bincode::decode_from_reader(&mut buf, config)?)
    }

    fn load_from_cargo_manifest_dir() -> anyhow::Result<Self> {
        let config = config::standard();

        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("..");
        path.push("asset.bin");

        let file = fs::File::open(&path).with_context(|| format!("{:?}", path.canonicalize()))?;
        let mut buf = BufReader::new(file);

        Ok(bincode::decode_from_reader(&mut buf, config)?)
    }
}
