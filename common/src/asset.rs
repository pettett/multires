use std::{
    fs,
    io::{BufReader, Write},
};

use bincode::{config, de, enc};
#[derive(Debug)]
pub enum ErrorKind {
    EncodeError(bincode::error::EncodeError),
    DecodeError(bincode::error::DecodeError),
    Io(std::io::Error),
}

impl From<bincode::error::EncodeError> for ErrorKind {
    fn from(value: bincode::error::EncodeError) -> Self {
        ErrorKind::EncodeError(value)
    }
}
impl From<bincode::error::DecodeError> for ErrorKind {
    fn from(value: bincode::error::DecodeError) -> Self {
        ErrorKind::DecodeError(value)
    }
}

impl From<std::io::Error> for ErrorKind {
    fn from(value: std::io::Error) -> Self {
        ErrorKind::Io(value)
    }
}

pub trait Asset: Sized + enc::Encode + de::Decode {
    fn save(&self) -> Result<(), ErrorKind> {
        let config = config::standard();

        let data = bincode::encode_to_vec(self, config)?;

        let mut file = fs::File::create("asset.bin")?;
        file.write_all(&data[..])?;

        Ok(())
    }
    fn load() -> Result<Self, ErrorKind> {
        let config = config::standard();

        let file = fs::File::open("asset.bin")?;
        let mut buf = BufReader::new(file);

        Ok(bincode::decode_from_reader(&mut buf, config)?)
    }
}
