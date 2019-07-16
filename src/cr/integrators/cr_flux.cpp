#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../mesh/mesh.hpp"
#include "../cr.hpp"
#include "cr_integrators.hpp"
#include "../../coordinates/coordinates.hpp"

// HLLE Riemann Solver to compute CR Flux
void CRIntegrator::CRFlux(int fdir, int k, int j, int il, int iu,
                          AthenaArray<Real> &w_l,    AthenaArray<Real> &w_r,
                          AthenaArray<Real> &vl,     AthenaArray<Real> &vr,
                          AthenaArray<Real> &crptl,  AthenaArray<Real> &crptr,
                          AthenaArray<Real> &vdif_l, AthenaArray<Real> &vdif_r,
                          AthenaArray<Real> &flx)
{
  Real vmax = pmy_cr->vmax; // Effective Speed of light
  Real *pcrpt_l, *pcrpt_r;

  if (fdir == CRF1) {
  	pcrpt_l = &(crptl(PC11,0));
  	pcrpt_r = &(crptr(PC11,0));
  } else if (fdir == CRF2) {
  	pcrpt_l = &(crptl(PC22,0));
  	pcrpt_r = &(crptr(PC22,0));
  } else if (fdir == CRF3) {
 	  pcrpt_l = &(crptl(PC33,0));
  	pcrpt_r = &(crptr(PC33,0));
  }

#pragma omp simd
  for (int i=il; i<=iu; ++i){
    Real meanadv = 0.5*(vl(i) + vr(i));
    Real meandiffv = 0.5*(vdif_l(i) + vdif_r(i));

    // Compute the max/min wave speeds
    Real al = std::min((meanadv - meandiffv),(vl(i)-vdif_l(i)));
    Real ar = std::max((meanadv + meandiffv),(vr(i)+vdif_r(i)));
    ar = std::min(ar,vmax * sqrt(pcrpt_r[i]));
    al = std::max(al,-vmax * sqrt(pcrpt_l[i]));

    Real bp = ar > 0.0 ? ar : 0.0;
    Real bm = al < 0.0 ? al : 0.0;

    // computer L/R fluxes along the lines bm/bp:
    // F_L - (S_L)U_L
    // F_R - (S_R)U_R

    Real fl_e = vmax * w_l(fdir,i) - bm * w_l(CRE,i);
    Real fr_e = vmax * w_r(fdir,i) - bp * w_r(CRE,i);
    Real fl_f1, fr_f1, fl_f2, fr_f2, fl_f3, fr_f3;

    if(fdir == CRF1){
      fl_f1 = vmax * crptl(PC11,i) * w_l(CRE,i) - bm * w_l(CRF1,i);
      fr_f1 = vmax * crptr(PC11,i) * w_r(CRE,i) - bp * w_r(CRF1,i);

      fl_f2 = vmax * crptl(PC12,i) * w_l(CRE,i) - bm * w_l(CRF2,i);
      fr_f2 = vmax * crptr(PC12,i) * w_r(CRE,i) - bp * w_r(CRF2,i);

      fl_f3 = vmax * crptl(PC13,i) * w_l(CRE,i) - bm * w_l(CRF3,i);
      fr_f3 = vmax * crptr(PC13,i) * w_r(CRE,i) - bp * w_r(CRF3,i);
    } else if (fdir == CRF2){
      fl_f1 = vmax * crptl(PC12,i) * w_l(CRE,i) - bm * w_l(CRF1,i);
      fr_f1 = vmax * crptr(PC12,i) * w_r(CRE,i) - bp * w_r(CRF1,i);

      fl_f2 = vmax * crptl(PC22,i) * w_l(CRE,i) - bm * w_l(CRF2,i);
      fr_f2 = vmax * crptr(PC22,i) * w_r(CRE,i) - bp * w_r(CRF2,i);

      fl_f3 = vmax * crptl(PC23,i) * w_l(CRE,i) - bm * w_l(CRF3,i);
      fr_f3 = vmax * crptr(PC23,i) * w_r(CRE,i) - bp * w_r(CRF3,i);

    } else if (fdir == CRF3){
      fl_f1 = vmax * crptl(PC13,i) * w_l(CRE,i) - bm * w_l(CRF1,i);
      fr_f1 = vmax * crptr(PC13,i) * w_r(CRE,i) - bp * w_r(CRF1,i);

      fl_f2 = vmax * crptl(PC23,i) * w_l(CRE,i) - bm * w_l(CRF2,i);
      fr_f2 = vmax * crptr(PC23,i) * w_r(CRE,i) - bp * w_r(CRF2,i);

      fl_f3 = vmax * crptl(PC33,i) * w_l(CRE,i) - bm * w_l(CRF3,i);
      fr_f3 = vmax * crptr(PC33,i) * w_r(CRE,i) - bp * w_r(CRF3,i);

    }

    // Compute the HLLE flux at interface
    Real tmp = 0.0;
    if (fabs(bm-bp) > TINY_NUMBER) tmp = 0.5*(bp + bm)/(bp - bm);

    flx(CRE,i)  = 0.5*(fl_e  + fr_e)  + (fl_e  - fr_e)  * tmp;
    flx(CRF1,i) = 0.5*(fl_f1 + fr_f1) + (fl_f1 - fr_f1) * tmp;
    flx(CRF2,i) = 0.5*(fl_f2 + fr_f2) + (fl_f2 - fr_f2) * tmp;
    flx(CRF3,i) = 0.5*(fl_f3 + fr_f3) + (fl_f3 - fr_f3) * tmp;

  } // end i
}
