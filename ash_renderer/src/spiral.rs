enum Leg {
    One,
    Two,
    Three,
    Four,
}

pub struct Spiral {
    layer: isize,
    leg: Leg,
    x: isize,
    y: isize,
}

impl Default for Spiral {
    fn default() -> Self {
        Self {
            layer: 1,
            leg: Leg::One,
            x: -1,
            y: 0,
        }
    }
}

impl Iterator for Spiral {
    type Item = (isize, isize);

    fn next(&mut self) -> Option<Self::Item> {
        match self.leg {
            Leg::One => {
                self.x += 1;
                if self.x == self.layer {
                    self.leg = Leg::Two;
                }
            }
            Leg::Two => {
                self.y += 1;
                if self.y == self.layer {
                    self.leg = Leg::Three;
                }
            }
            Leg::Three => {
                self.x -= 1;
                if -self.x == self.layer {
                    self.leg = Leg::Four;
                }
            }
            Leg::Four => {
                self.y -= 1;
                if -self.y == self.layer {
                    self.leg = Leg::One;
                    self.layer += 1;
                }
            }
        };
        Some((self.x, self.y))
    }
}
