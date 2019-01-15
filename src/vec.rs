use packed_simd::f32x16;

#[derive(Debug, PartialEq, Default, Clone, Copy)]
pub struct Vec3dx16 {
    pub xs: f32x16,
    pub ys: f32x16,
    pub zs: f32x16,
}

impl Vec3dx16 {
    pub fn from_tuple((x, y, z): (f32, f32, f32)) -> Vec3dx16 {
        Vec3dx16 {
            xs: f32x16::splat(x),
            ys: f32x16::splat(y),
            zs: f32x16::splat(z),
        }
    }
}

macro_rules! impl_op {
    ($op_name:ident, $f:ident, $op_assign_name:ident, $f_assign:ident, $op: tt, $op_assign:tt) => {
        impl std::ops::$op_name for Vec3dx16 {
            type Output = Vec3dx16;

            fn $f(self, other: Vec3dx16) -> Vec3dx16 {
                Vec3dx16 {
                    xs: self.xs $op other.xs,
                    ys: self.ys $op other.ys,
                    zs: self.zs $op other.zs,
                }
            }
        }

        impl <'a> std::ops::$op_name<Vec3dx16> for &'a Vec3dx16 {
            type Output = Vec3dx16;

            fn $f(self, other: Vec3dx16) -> Vec3dx16 {
                Vec3dx16 {
                    xs: self.xs $op other.xs,
                    ys: self.ys $op other.ys,
                    zs: self.zs $op other.zs,
                }
            }
        }

        impl <'a> std::ops::$op_name<&'a Vec3dx16> for Vec3dx16 {
            type Output = Vec3dx16;

            fn $f(self, other: &Vec3dx16) -> Vec3dx16 {
                Vec3dx16 {
                    xs: self.xs $op other.xs,
                    ys: self.ys $op other.ys,
                    zs: self.zs $op other.zs,
                }
            }
        }

        impl <'a, 'b> std::ops::$op_name<&'a Vec3dx16> for &'b Vec3dx16 {
            type Output = Vec3dx16;

            fn $f(self, other: &Vec3dx16) -> Vec3dx16 {
                Vec3dx16 {
                    xs: self.xs $op other.xs,
                    ys: self.ys $op other.ys,
                    zs: self.zs $op other.zs,
                }
            }
        }

        impl std::ops::$op_assign_name for Vec3dx16 {
            fn $f_assign(&mut self, other: Self) {
                self.xs $op_assign other.xs;
                self.ys $op_assign other.ys;
                self.zs $op_assign other.zs;
            }
        }
    }
}

impl_op!(Add, add, AddAssign, add_assign, +, +=);
impl_op!(Sub, sub, SubAssign, sub_assign, -, -=);
impl_op!(Mul, mul, MulAssign, mul_assign, *, *=);
impl_op!(Div, div, DivAssign, div_assign, /, /=);
