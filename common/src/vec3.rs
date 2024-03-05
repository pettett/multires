#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Vec3(pub(crate) glam::Vec3);

impl Into<Vec3> for glam::Vec3 {
    fn into(self) -> Vec3 {
        Vec3(self)
    }
}
impl Into<Vec3> for glam::Vec3A {
    fn into(self) -> Vec3 {
        Vec3(self.into())
    }
}
impl Into<glam::Vec3> for Vec3 {
    fn into(self) -> glam::Vec3 {
        self.0
    }
}

impl bincode::Encode for Vec3 {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.0.x, encoder)?;
        bincode::Encode::encode(&self.0.y, encoder)?;
        bincode::Encode::encode(&self.0.z, encoder)?;
        Ok(())
    }
}

impl bincode::Decode for Vec3 {
    fn decode<D: bincode::de::Decoder>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self(glam::Vec3 {
            x: bincode::Decode::decode(decoder)?,
            y: bincode::Decode::decode(decoder)?,
            z: bincode::Decode::decode(decoder)?,
        }))
    }
}
impl bincode::BorrowDecode<'_> for Vec3 {
    fn borrow_decode<D: bincode::de::Decoder>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self(glam::Vec3 {
            x: bincode::Decode::decode(decoder)?,
            y: bincode::Decode::decode(decoder)?,
            z: bincode::Decode::decode(decoder)?,
        }))
    }
}
