use std::borrow::Borrow;
use std::marker::PhantomData;

use crate::vec::Vec3dx16;
use packed_simd::{f32x16, m32x16, u8x16, FromCast};
use std::f32::{INFINITY, NEG_INFINITY};

const EPSILON: f32 = 1e-2;
const MAX_ITERATIONS: usize = 30;

pub fn norm(v: &Vec3dx16) -> f32x16 {
    (v.xs * v.xs + v.ys * v.ys + v.zs * v.zs).sqrt()
}

pub trait World: Send + Sync {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16;
}

pub enum Axis {
    X, Y, Z
}

pub struct Rotation<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    pub around: Axis,
    pub inner: T,
    pub angle: f32,
    marker: PhantomData<TBor>
}

pub type RotRef<'a, T> = Rotation<&'a T, T>;
pub type RotT<T> = Rotation<T, T>;

impl <T, TBor> Rotation<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    pub fn new(inner: T, around: Axis, angle: f32) -> Rotation<T, TBor> {
        Rotation {
            inner, around, angle, marker: PhantomData
        }
    }
}

impl <T, TBor> World for Rotation<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        let acos = self.angle.cos();
        let asin = self.angle.sin();
        let mut x_ = x.clone();

        // TODO: Make sure positive direction is consistent here
        match self.around {
            Axis::X => {
                x_.ys = x.ys * f32x16::splat(acos) - x.zs * f32x16::splat(asin);
                x_.zs = x.ys * f32x16::splat(asin) + x.zs * f32x16::splat(acos);
            }
            Axis::Y => {
                x_.xs = x.xs * f32x16::splat(acos) - x.zs * f32x16::splat(asin);
                x_.zs = x.xs * f32x16::splat(asin) + x.zs * f32x16::splat(acos);
            }
            Axis::Z => {
                x_.xs = x.xs * f32x16::splat(acos) - x.ys * f32x16::splat(asin);
                x_.ys = x.xs * f32x16::splat(asin) + x.ys * f32x16::splat(acos);
            }
        }
        self.inner.borrow().distance_estimator(&x_)
    }
}

pub struct Translation<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    pub inner: T,
    pub at: (f32, f32, f32),
    marker: PhantomData<TBor>
}

pub type TransRef<'a, T> = Translation<&'a T, T>;
pub type TransT<T> = Translation<T, T>;

impl <T, TBor> Translation<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    pub fn new(inner: T, at: (f32, f32, f32)) -> Translation<T, TBor> {
        Translation {
            inner, at, marker: PhantomData
        }
    }
}

impl <T, TBor> World for Translation<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        // haha
        let splAT = Vec3dx16 {
            xs: f32x16::splat(self.at.0),
            ys: f32x16::splat(self.at.1),
            zs: f32x16::splat(self.at.2),
        };
        self.inner.borrow().distance_estimator(&(x - splAT))
    }
}

pub struct Scale<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    pub inner: T,
    pub scaling: (f32, f32, f32),
    marker: PhantomData<TBor>
}

pub type ScaleRef<'a, T> = Scale<&'a T, T>;
pub type ScaleT<T> = Scale<T, T>;

impl <T, TBor> Scale<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    pub fn new(inner: T, scaling: (f32, f32, f32)) -> Scale<T, TBor> {
        Scale {
            inner, scaling, marker: PhantomData
        }
    }
}

impl <T, TBor> World for Scale<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        // haha
        let splat = Vec3dx16 {
            xs: f32x16::splat(self.scaling.0),
            ys: f32x16::splat(self.scaling.1),
            zs: f32x16::splat(self.scaling.2),
        };
        self.inner.borrow().distance_estimator(&(x / splat))
            * self.scaling.0.min(self.scaling.1).min(self.scaling.2)
    }
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


pub struct UnitSphere;

impl World for UnitSphere {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        norm(x) - f32x16::splat(1.)
    }
}

pub struct UnitCube;

impl World for UnitCube {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        let xs = x.xs.abs() - f32x16::splat(1.);
        let ys = x.ys.abs() - f32x16::splat(1.);
        let zs = x.zs.abs() - f32x16::splat(1.);
        xs.max(ys).max(zs)
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

pub fn construct_cuboid(at: (f32, f32, f32), dims: (f32, f32, f32)) -> TransT<ScaleT<UnitCube>> {
    let cu = UnitCube;
    let scaled = Scale::new(cu, dims);
    let translated = Translation::new(scaled, at);
    translated
}

pub fn construct_sphere(at: (f32, f32, f32), rad: f32) -> TransT<ScaleT<UnitSphere>> {
    let sp = UnitSphere;
    let scaled = Scale::new(sp, (rad, rad, rad));
    let translated = Translation::new(scaled, at);
    translated
}

pub fn raymarch(world: &dyn World, mut poses: Vec3dx16, dirs: Vec3dx16) -> u8x16 {
    let norms = norm(&dirs);
    let dirs = dirs
        / Vec3dx16 {
            xs: norms,
            ys: norms,
            zs: norms,
        };

    let mut res = f32x16::splat(0.);
    let mut hit = m32x16::splat(false);

    let mut last_des = f32x16::splat(0.);

    for i in 0..MAX_ITERATIONS {
        if hit.all() {
            break;
        }
        let des = world.distance_estimator(&poses);
        // Check for collisions (eg. very small distance estimates)

        let rays_hit = des.le(f32x16::splat(EPSILON)) & des.lt(last_des);
        let new_hits = rays_hit & !hit;

        if new_hits.any() {
            let fmask = -f32x16::from_cast(new_hits);

            let colors = 255. - 255. * (des / last_des).min(f32x16::splat(1.));

            res = fmask * colors + (1. - fmask) * res;

            hit |= new_hits;
        }
        if i == MAX_ITERATIONS - 1 {
            let fmask = -f32x16::from_cast(!hit);

            let colors = 255. - 255. * (des / last_des).min(f32x16::splat(1.));

            res = fmask * colors + (1. - fmask) * res;
        }


        let move_vec = Vec3dx16 {
            xs: dirs.xs * des,
            ys: dirs.ys * des,
            zs: dirs.zs * des,
        };
        poses += move_vec;
        last_des = des;
    }
    u8x16::from_cast(res)
}
