use crate::vec::Vec3dx16;
use packed_simd::{f32x16, u8x16};

const EPSILON: f32 = 5e-2;

pub fn norm(v: &Vec3dx16) -> f32x16 {
    (v.xs * v.xs + v.ys * v.ys + v.zs * v.zs).sqrt()
}

pub trait World: Send + Sync {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16;
}

pub struct Sphere {
    pub pos: (f32, f32, f32),
    pub size: f32,
}

impl World for Sphere {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        norm(&(x - Vec3dx16::from_tuple(self.pos))) - f32x16::splat(self.size)
    }
}

// pub struct Cube {
//     pub center: Array1<f64>,
//     pub dims: Array1<f64>
// }

// impl World for Cube {
//     fn distance_estimator(&self, x: &Array1<f64>) -> f64 {
//         let delta = x - &self.center;
//         let delta = delta.map(|x| x.abs());

//         (delta - &self.dims).iter().fold(0., |a: f64, &b| a.max(b))
//     }
// }

pub fn raymarch(world: &dyn World, mut poses: Vec3dx16, dirs: Vec3dx16) -> u8x16 {
    let norms = norm(&dirs);
    let dirs = dirs
        / Vec3dx16 {
            xs: norms,
            ys: norms,
            zs: norms,
        };

    let mut res = u8x16::splat(0);
    let mut hit = 0i16;

    let mut last_des = f32x16::splat(0.);

    for i in 0..30 {
        let des = world.distance_estimator(&poses);
        // Check for small
        for i in 0..16 {
            if (hit >> i) & 1 != 0 {
                continue;
            }

            let de = des.extract(i);
            let last = last_des.extract(i);

            if de <= EPSILON {
                if de < last {
                    res = res.replace(i, (255. - 255. * de / last) as u8);
                    hit |= 1 << i;
                }
            }
            if de > 40. {
                hit |= 1 << i;
            }
        }
        let des_vec = Vec3dx16 {
            xs: des,
            ys: des,
            zs: des,
        };
        poses += dirs * des_vec;
        last_des = des;
    }
    res
}
