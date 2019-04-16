use std::borrow::Borrow;
use std::marker::PhantomData;

use crate::vec::Vec3dx16;
use packed_simd::{u32x16, f32x16, m32x16, FromCast};
use std::f32::{INFINITY, NEG_INFINITY};

const EPSILON: f32 = 1e-2;
const MAX_ITERATIONS: usize = 30;

pub fn norm(v: &Vec3dx16) -> f32x16 {
    (v.xs * v.xs + v.ys * v.ys + v.zs * v.zs).sqrt()
}

pub trait World: Send + Sync {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16;

    fn color(&self, x: &Vec3dx16) -> Vec3dx16;
}

pub enum Axis {
    X, Y, Z
}

pub struct Checkers<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    pub inner: T,
    pub color1: (f32, f32, f32),
    pub color2: (f32, f32, f32),
    marker: PhantomData<TBor>
}

pub type CheckerRef<'a, T> = Checkers<&'a T, T>;
pub type CheckerT<T> = Checkers<T, T>;

impl <T, TBor> Checkers<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    pub fn new(inner: T, color1: (f32, f32, f32), color2: (f32, f32, f32)) -> Checkers<T, TBor> {
        Checkers {
            inner, color1, color2, marker: PhantomData
        }
    }
}

impl <T, TBor> World for Checkers<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        // haha
        self.inner.borrow().distance_estimator(x)
    }

    fn color(&self, x: &Vec3dx16) -> Vec3dx16 {
        let xs = u32x16::from_cast(x.xs.abs() + f32x16::splat(0.5));
        let ys = u32x16::from_cast(x.ys.abs() + f32x16::splat(0.5));
        let zs = u32x16::from_cast(x.zs.abs() + f32x16::splat(0.5));

        let which = ((xs + ys + zs) % 2).eq(u32x16::splat(0));
        let mask = Vec3dx16::splat(-f32x16::from_cast(which));

        mask * Vec3dx16::from_tuple(self.color1)
            + (Vec3dx16::from_tuple((1., 1., 1.,)) - mask) * Vec3dx16::from_tuple(self.color2)
    }
}

pub struct Coloring<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    pub inner: T,
    pub color: (f32, f32, f32),
    marker: PhantomData<TBor>
}

pub type ColorRef<'a, T> = Coloring<&'a T, T>;
pub type ColorT<T> = Coloring<T, T>;

impl <T, TBor> Coloring<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    pub fn new(inner: T, color: (f32, f32, f32)) -> Coloring<T, TBor> {
        Coloring {
            inner, color, marker: PhantomData
        }
    }
}

impl <T, TBor> World for Coloring<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        // haha
        self.inner.borrow().distance_estimator(x)
    }

    fn color(&self, x: &Vec3dx16) -> Vec3dx16 {
        Vec3dx16::from_tuple(self.color)
    }
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

    fn transform(&self, x: &Vec3dx16) -> Vec3dx16 {
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
        x_
    }
}

impl <T, TBor> World for Rotation<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        self.inner.borrow().distance_estimator(&self.transform(x))
    }

    fn color(&self, x: &Vec3dx16) -> Vec3dx16 {
        self.inner.borrow().color(&self.transform(x))
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

    fn transform(&self, x: &Vec3dx16) -> Vec3dx16 {
        // haha
        let splAT = Vec3dx16 {
            xs: f32x16::splat(self.at.0),
            ys: f32x16::splat(self.at.1),
            zs: f32x16::splat(self.at.2),
        };

        x - splAT
    }
}

impl <T, TBor> World for Translation<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        self.inner.borrow().distance_estimator(&self.transform(x))
    }
    fn color(&self, x: &Vec3dx16) -> Vec3dx16 {
        self.inner.borrow().color(&self.transform(x))
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

    fn transform(&self, x: &Vec3dx16) -> Vec3dx16 {
        let splat = Vec3dx16 {
            xs: f32x16::splat(self.scaling.0),
            ys: f32x16::splat(self.scaling.1),
            zs: f32x16::splat(self.scaling.2),
        };

        x / splat
    }
}

impl <T, TBor> World for Scale<T, TBor>
    where
        T: Borrow<TBor> + Send + Sync,
        TBor: World
{
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        self.inner.borrow().distance_estimator(&self.transform(x))
            * self.scaling.0.min(self.scaling.1).min(self.scaling.2)
    }
    fn color(&self, x: &Vec3dx16) -> Vec3dx16 {
        self.inner.borrow().color(&self.transform(x))
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

    fn color(&self, x: &Vec3dx16) -> Vec3dx16 {
        let mut distances = f32x16::splat(INFINITY);
        let mut colors = Vec3dx16::from_tuple((0., 1., 0.));

        for obj in &self.objects {
            let distances_ = obj.distance_estimator(x);
            let closer = distances_.lt(distances);

            if closer.any() {
                let mask = -f32x16::from_cast(closer);

                let colors_ = obj.color(x);
                colors.xs = mask * colors_.xs + (1. - mask) * colors.xs;
                colors.ys = mask * colors_.ys + (1. - mask) * colors.ys;
                colors.zs = mask * colors_.zs + (1. - mask) * colors.zs;
            }

            distances = distances.min(distances_);
        }
        colors
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

    fn color(&self, x: &Vec3dx16) -> Vec3dx16 {
        self.objects[0].color(x)
    }
}


pub struct UnitSphere;

impl World for UnitSphere {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        norm(x) - f32x16::splat(1.)
    }

    fn color(&self, x: &Vec3dx16) -> Vec3dx16 {
        Vec3dx16::from_tuple((1., 1., 1.,))
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
    fn color(&self, x: &Vec3dx16) -> Vec3dx16 {
        Vec3dx16::from_tuple((1., 1., 1.,))
    }
}

pub struct Plane {
    pub height: f32
}

impl World for Plane {
    fn distance_estimator(&self, x: &Vec3dx16) -> f32x16 {
        (x.ys - f32x16::splat(self.height)).abs()
    }
    fn color(&self, x: &Vec3dx16) -> Vec3dx16 {
        Vec3dx16::from_tuple((1., 1., 1.,))
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

pub fn raymarch(world: &dyn World, mut poses: Vec3dx16, dirs: Vec3dx16) -> Vec3dx16 {
    let norms = norm(&dirs);
    let dirs = dirs
        / Vec3dx16 {
            xs: norms,
            ys: norms,
            zs: norms,
        };

    let mut res = Vec3dx16::from_tuple((0., 0., 0.));
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

            let gray = (des / last_des).min(f32x16::splat(1.));

            let colors = world.color(&poses) * Vec3dx16::splat(1. - gray);

            res.xs = fmask * colors.xs + (1. - fmask) * res.xs;
            res.ys = fmask * colors.ys + (1. - fmask) * res.ys;
            res.zs = fmask * colors.zs + (1. - fmask) * res.zs;

            hit |= new_hits;
        }
        if i == MAX_ITERATIONS - 1 {
            let fmask = -f32x16::from_cast(!hit);

            let gray = (des / last_des).min(f32x16::splat(1.));

            let colors = world.color(&poses) * Vec3dx16::splat(1. - gray);

            res.xs = fmask * colors.xs + (1. - fmask) * res.xs;
            res.ys = fmask * colors.ys + (1. - fmask) * res.ys;
            res.zs = fmask * colors.zs + (1. - fmask) * res.zs;
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
