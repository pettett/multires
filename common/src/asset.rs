use std::{fs, io::Write};
#[derive(Debug)]
pub enum ErrorKind {
    Bincode(Box<bincode::ErrorKind>),
    Io(std::io::Error),
}

impl From<Box<bincode::ErrorKind>> for ErrorKind {
    fn from(value: Box<bincode::ErrorKind>) -> Self {
        ErrorKind::Bincode(value)
    }
}
impl From<std::io::Error> for ErrorKind {
    fn from(value: std::io::Error) -> Self {
        ErrorKind::Io(value)
    }
}

pub trait Asset: Sized + serde::Serialize + serde::de::DeserializeOwned {
    fn save(&self) -> Result<(), Box<bincode::ErrorKind>> {
        let data = bincode::serialize(self)?;

        let mut file = fs::File::create("asset.bin")?;
        file.write_all(&data[..])?;

        Ok(())
    }
    fn load() -> Result<Self, ErrorKind> {
        let mut file = fs::File::open("asset.bin")?;

        Ok(bincode::deserialize_from(&mut file)?)
    }
}
