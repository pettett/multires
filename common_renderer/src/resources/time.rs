use bevy_ecs::system::Resource;

#[derive(Resource, Default)]
pub struct Time {
    pub delta_time: f32,
    pub time: f32,
}

impl Time {
    pub fn tick(&mut self, delta_time: f32) {
        self.delta_time = delta_time;
        self.time += delta_time;
    }
}
