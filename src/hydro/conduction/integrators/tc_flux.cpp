// Athena++ headers
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../mesh/mesh.hpp"
#include "../tc.hpp"
#include "tc_integrators.hpp"
#include "../../../coordinates/coordinates.hpp"

// The four independent variables are:
// e, (T/e)F_1, (T/e)F_2, (T/e)F_3
// But the stored variables are
// e, F_1, F_2, F_3
// The fluxes are:
// V_m (F_1, F_2, F_3)
// V_m T * (Unity Tensor), all off-diagonal components are zero

void TCIntegrator::TCFlux(int fdir, int il, int iu,
                          AthenaArray<Real> &t_l, AthenaArray<Real> &t_r,
                          AthenaArray<Real> &rho_l, AthenaArray<Real> &rho_r,
                          AthenaArray<Real> &w_l, AthenaArray<Real> &w_r,
                          AthenaArray<Real> &vdiff_l, AthenaArray<Real> &vdiff_r,
                          AthenaArray<Real> &flx)
{
  Real vmax = pmy_tc->vmax;
#pragma omp simd
  for (int i=il; i<=iu; ++i) {

    Real meandiffv = 0.5*(vdiff_l(i)+vdiff_r(i));

    Real meanrho = 0.5*(rho_l(i)+rho_r(i));
    Real rhoratiol=meanrho/rho_l(i);
    Real rhoratior=meanrho/rho_r(i);

    Real al = std::min(meandiffv,vdiff_l(i));
    Real ar = std::max(meandiffv,vdiff_r(i));

    Real bp = ar > 0.0 ? ar : 0.0;
    Real bm = al < 0.0 ? al : 0.0;

    // computer L/R fluxes along lines
    // F_L - (S_L)U_L
    // F_R - (S_R)U_R
    Real fl_e = vmax * w_l(fdir,i) - bm * w_l(0,i)*rhoratiol;
    Real fr_e = vmax * w_r(fdir,i) - bp * w_r(0,i)*rhoratior;
    Real fl_f1, fr_f1, fl_f2, fr_f2, fl_f3, fr_f3;

    if (fdir == 1) {

      fl_f1 = vmax * t_l(i) - bm * w_l(1,i)/meanrho;
      fr_f1 = vmax * t_r(i) - bp * w_r(1,i)/meanrho;
      fl_f2 = 0.0;
      fr_f2 = 0.0;
      fl_f3 = 0.0;
      fr_f3 = 0.0;

    } else if (fdir == 2) {

      fl_f1 = 0.0;
      fr_f1 = 0.0;
      fl_f2 = vmax * t_l(i) - bm * w_l(2,i)/meanrho;
      fr_f2 = vmax * t_r(i) - bp * w_r(2,i)/meanrho;
      fl_f3 = 0.0;
      fr_f3 = 0.0;

    } else if (fdir == 3) {

      fl_f1 = 0.0;
      fr_f1 = 0.0;
      fl_f2 = 0.0;
      fr_f2 = 0.0;
      fl_f3 = vmax * t_l(i) - bm * w_l(3,i)/meanrho;
      fr_f3 = vmax * t_r(i) - bp * w_r(3,i)/meanrho;

    }

    // Calculate the HLLE flux
    Real tmp = 0.0;
    if (fabs(bm-bp) > TINY_NUMBER) {
    	tmp = 0.5*(bp + bm)/(bp - bm);
    }

    flx(0,i) = 0.5*(fl_e + fr_e) + (fl_e - fr_e) * tmp;
    flx(1,i) = 0.5*(fl_f1 + fr_f1) + (fl_f1 - fr_f1) * tmp;
    flx(2,i) = 0.5*(fl_f2 + fr_f2) + (fl_f2 - fr_f2) * tmp;
    flx(3,i) = 0.5*(fl_f3 + fr_f3) + (fl_f3 - fr_f3) * tmp;

  }

  return;
}
