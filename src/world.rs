use std::f32::{INFINITY, NEG_INFINITY};
use crate::vec::Vec3dx16;
use packed_simd::{f32x16, u8x16};

const EPSILON: f32 = 1e-2;

pub fn norm(v: &Vec3dx16) -> f32x16 {
    (v.xs * v.xs + v.ys * v.ys + v.zs * v.zs).sqrt()
}

pub trait World: Send + Sync {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16;
}

pub struct Union {
    pub objects: Vec<Box<dyn World>>
}

impl World for Union {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        let mut res = f32x16::splat(INFINITY);
        for obj in &self.objects {
            res = res.min(obj.distance_estimator(x));
        }
        res
    }
}

pub struct Intersection {
    pub objects: Vec<Box<dyn World>>
}

impl World for Intersection {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        let mut res = f32x16::splat(NEG_INFINITY);
        for obj in &self.objects {
            res = res.max(obj.distance_estimator(x));
        }
        res
    }
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

pub struct Cube {
    pub center: (f32, f32, f32),
    pub dims: (f32, f32, f32),
}

impl World for Cube {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        let delta = x - Vec3dx16::from_tuple(self.center);
        let delta = Vec3dx16 {
            xs: delta.xs.abs(),
            ys: delta.ys.abs(),
            zs: delta.zs.abs(),
        };

        let delta = delta - Vec3dx16::from_tuple(self.dims);

        delta.xs.max(delta.ys).max(delta.zs)
    }
}

pub struct Plane {
    pub height: f32
}

impl World for Plane {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        (x.ys - f32x16::splat(self.height)).abs()
    }
}

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

    for _ in 0..100 {
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
        }

        let move_vec = Vec3dx16 {
            xs: dirs.xs * des,
            ys: dirs.ys * des,
            zs: dirs.zs * des,
        };
        poses += move_vec;
        last_des = des;
    }
    res
}
