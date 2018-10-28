// C++ headers
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../mesh/mesh.hpp"
#include "../cr.hpp"
#include "cr_integrators.hpp"
#include "../../coordinates/coordinates.hpp"

// HLLE Flux for CR Transport
void CRIntegrator::CRFlux(int fdir, int k, int j, int il, int iu, 
                          AthenaArray<Real> &w_l, AthenaArray<Real> &w_r, 
                          AthenaArray<Real> &vl, AthenaArray<Real> &vr, 
                          AthenaArray<Real> &eddl, AthenaArray<Real> &eddr,
                          AthenaArray<Real> &vdiff_l, AthenaArray<Real> &vdiff_r, 
                          AthenaArray<Real> &flx) {
  // First, get the photon diffusion speed
  Real vmax = pmy_cr->vmax;
  Real *pedd_l, *pedd_r;
  
  if (fdir == CRF1) {
  	pedd_l = &(eddl(PC11,0));
  	pedd_r = &(eddr(PC11,0));
  } else if (fdir == CRF2) {
  	pedd_l = &(eddl(PC22,0));
  	pedd_r = &(eddr(PC22,0));  	
  } else if (fdir == CRF3) {
 	pedd_l = &(eddl(PC33,0));
  	pedd_r = &(eddr(PC33,0));
  }

  // use sigma_l, sigma_r to get diffusion speed
#pragma omp simd
  for (int i=il; i<=iu; ++i){
    Real meanadv=0.5*(vl(i) + vr(i));
    Real meandiffv = 0.5*(vdiff_l(i)+vdiff_r(i));

    Real al = std::min((meanadv - meandiffv),(vl(i)-vdiff_l(i)));
    Real ar = std::max((meanadv + meandiffv),(vr(i)+vdiff_r(i)));
    ar = std::min(ar,vmax * sqrt(pedd_r[i]));
    al = std::max(al,-vmax * sqrt(pedd_l[i]));

    Real bp = ar > 0.0 ? ar : 0.0;
    Real bm = al < 0.0 ? al : 0.0;

    // computer L/R fluxes along lines
    // F_L - (S_L)U_L
    // F_R - (S_R)U_R

    Real fl_e = vmax * w_l(fdir,i) - bm * w_l(CRE,i);
    Real fr_e = vmax * w_r(fdir,i) - bp * w_r(CRE,i);
    Real fl_f1, fr_f1, fl_f2, fr_f2, fl_f3, fr_f3;
    
    if(fdir == CRF1){
      fl_f1 = vmax * eddl(PC11,i) * w_l(CRE,i) - bm * w_l(CRF1,i);
      fr_f1 = vmax * eddr(PC11,i) * w_r(CRE,i) - bp * w_r(CRF1,i);

      fl_f2 = vmax * eddl(PC12,i) * w_l(CRE,i) - bm * w_l(CRF2,i);
      fr_f2 = vmax * eddr(PC12,i) * w_r(CRE,i) - bp * w_r(CRF2,i);

      fl_f3 = vmax * eddl(PC13,i) * w_l(CRE,i) - bm * w_l(CRF3,i);
      fr_f3 = vmax * eddr(PC13,i) * w_r(CRE,i) - bp * w_r(CRF3,i);
    } else if(fdir == CRF2){
      fl_f1 = vmax * eddl(PC12,i) * w_l(CRE,i) - bm * w_l(CRF1,i);
      fr_f1 = vmax * eddr(PC12,i) * w_r(CRE,i) - bp * w_r(CRF1,i);

      fl_f2 = vmax * eddl(PC22,i) * w_l(CRE,i) - bm * w_l(CRF2,i);
      fr_f2 = vmax * eddr(PC22,i) * w_r(CRE,i) - bp * w_r(CRF2,i);

      fl_f3 = vmax * eddl(PC23,i) * w_l(CRE,i) - bm * w_l(CRF3,i);
      fr_f3 = vmax * eddr(PC23,i) * w_r(CRE,i) - bp * w_r(CRF3,i);

    }else if(fdir == CRF3){
      fl_f1 = vmax * eddl(PC13,i) * w_l(CRE,i) - bm * w_l(CRF1,i);
      fr_f1 = vmax * eddr(PC13,i) * w_r(CRE,i) - bp * w_r(CRF1,i);

      fl_f2 = vmax * eddl(PC23,i) * w_l(CRE,i) - bm * w_l(CRF2,i);
      fr_f2 = vmax * eddr(PC23,i) * w_r(CRE,i) - bp * w_r(CRF2,i);

      fl_f3 = vmax * eddl(PC33,i) * w_l(CRE,i) - bm * w_l(CRF3,i);
      fr_f3 = vmax * eddr(PC33,i) * w_r(CRE,i) - bp * w_r(CRF3,i);

    }
  
    // Calculate the HLLE flux
    Real tmp = 0.0;
    if (fabs(bm-bp) > TINY_NUMBER) {tmp = 0.5*(bp + bm)/(bp - bm);}
    
    flx(CRE,i)  = 0.5*(fl_e  + fr_e)  + (fl_e  - fr_e)  * tmp;
    flx(CRF1,i) = 0.5*(fl_f1 + fr_f1) + (fl_f1 - fr_f1) * tmp;
    flx(CRF2,i) = 0.5*(fl_f2 + fr_f2) + (fl_f2 - fr_f2) * tmp;
    flx(CRF3,i) = 0.5*(fl_f3 + fr_f3) + (fl_f3 - fr_f3) * tmp;       

  } // end i
}
