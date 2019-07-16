#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../mesh/mesh.hpp"
#include "../cr.hpp"
#include "cr_integrators.hpp"
#include "../../coordinates/coordinates.hpp"

// Piecewise constant (donor cell) reconstruction methods
void CRIntegrator::DonorCellX1(int const k, int const j,
                               int const il, int const iu,
                               AthenaArray<Real> const &u_cr,
                               AthenaArray<Real> const &prim,
			                         AthenaArray<Real> const &crpt,
                               AthenaArray<Real> &w_l, AthenaArray<Real> &w_r,
                               AthenaArray<Real> &v_l, AthenaArray<Real> &v_r,
                               AthenaArray<Real> &crptl,
                               AthenaArray<Real> &crptr)
{
  // compute L/R states for each cosmic ray variable
  for (int n=0; n<NCR; ++n){
#pragma omp simd
    for (int i=il; i<=iu; ++i){
      w_l(n,i) = u_cr(n,k,j,i-1);
      w_r(n,i) = u_cr(n,k,j,i);
    }
  }

  // compute L/R states for the cosmic ray pressure tensor
  for (int n=0; n<6; ++n){
#pragma omp simd
    for (int i=il; i<=iu; ++i){
      crptl(n,i) = crpt(n,k,j,i-1);
      crptr(n,i) = crpt(n,k,j,i);
    }
  }

  // compute L/R states for the velocities
#pragma omp simd
  for (int i=il; i<=iu; ++i){
    v_l(i) = prim(IVX,k,j,i-1);
    v_r(i) = prim(IVX,k,j,i);
  }

  return;
}

void CRIntegrator::DonorCellX2(int const k, int const j,
                               int const il, int const iu,
			                         AthenaArray<Real> const &u_cr,
                               AthenaArray<Real> const &prim,
			                         AthenaArray<Real> const &crpt,
                               AthenaArray<Real> &w_l, AthenaArray<Real> &w_r,
                               AthenaArray<Real> &v_l, AthenaArray<Real> &v_r,
                               AthenaArray<Real> &crptl,
                               AthenaArray<Real> &crptr)
{
  // compute L/R states for each cosmic ray variable
  for (int n=0; n<NCR; ++n){
#pragma omp simd
    for (int i=il; i<=iu; ++i){
      w_l(n,i) = u_cr(n,k,j-1,i);
      w_r(n,i) = u_cr(n,k,j,i);
    }
  }

  // compute L/R states for the cosmic ray pressure tensor
  for (int n=0; n<6; ++n){
#pragma omp simd
    for (int i=il; i<=iu; ++i){
      crptl(n,i) = crpt(n,k,j-1,i);
      crptr(n,i) = crpt(n,k,j,i);
    }
  }

  // compute L/R states for the velocities
#pragma omp simd
  for (int i=il; i<=iu; ++i){
    v_l(i) = prim(IVY,k,j-1,i);
    v_r(i) = prim(IVY,k,j,i);
  }

  return;
}

void CRIntegrator::DonorCellX3(int const k, int const j,
                               int const il, int const iu,
			                         AthenaArray<Real> const &u_cr,
                               AthenaArray<Real> const &prim,
			                         AthenaArray<Real> const &crpt,
                               AthenaArray<Real> &w_l, AthenaArray<Real> &w_r,
                               AthenaArray<Real> &v_l, AthenaArray<Real> &v_r,
                               AthenaArray<Real> &crptl,
                               AthenaArray<Real> &crptr)
{
  // compute L/R states for each cosmic ray variable
  for (int n=0; n<NCR; ++n){
#pragma omp simd
    for (int i=il; i<=iu; ++i){
      w_l(n,i) = u_cr(n,k-1,j,i);
      w_r(n,i) = u_cr(n,k,j,i);
    }
  }

  // compute L/R states for the cosmic ray pressure tensor
  for (int n=0; n<6; ++n){
#pragma omp simd
    for (int i=il; i<=iu; ++i){
      crptl(n,i) = crpt(n,k-1,j,i);
      crptr(n,i) = crpt(n,k,j,i);
    }
  }

  // compute L/R states for the velocities
#pragma omp simd
  for (int i=il; i<=iu; ++i){
    v_l(i) = prim(IVZ,k-1,j,i);
    v_r(i) = prim(IVZ,k,j,i);
  }

  return;
}

// Piecewise Linear Reconstruction methods
void CRIntegrator::PieceWiseLinear(int const k, int const j,
                                   int const il, int const iu,
				                           AthenaArray<Real> &u_cr,
				                           AthenaArray<Real> &prim,
				                           AthenaArray<Real> &crpt,
                                   AthenaArray<Real> &w_l,
                                   AthenaArray<Real> &w_r,
                                   AthenaArray<Real> &v_l,
				                           AthenaArray<Real> &v_r,
                                   AthenaArray<Real> &crptl,
				                           AthenaArray<Real> &crptr,
                                   int dir)
{
  // Dummy pointers that are used to refer to data slices
  AthenaArray<Real> q, ql, qr;

  for (int n=0; n<NCR; ++n) {
    q.InitWithShallowSlice(u_cr,4,n,1);
    ql.InitWithShallowSlice(w_l,2,n,1);
    qr.InitWithShallowSlice(w_r,2,n,1);

    if (dir == X1DIR) {
      GetOneVariableX1(k,j,il,iu,q,ql,qr);
    } else if (dir == X2DIR) {
      GetOneVariableX2(k,j,il,iu,q,ql,qr);
    } else if (dir == X3DIR) {
      GetOneVariableX3(k,j,il,iu,q,ql,qr);
    }
  }

  // Cosmic Ray Pressure tensor
  for (int n=0; n<6; ++n) {
    q.InitWithShallowSlice(crpt,4,n,1);
    ql.InitWithShallowSlice(crptl,2,n,1);
    qr.InitWithShallowSlice(crptr,2,n,1);

    if (dir == X1DIR) {
      GetOneVariableX1(k,j,il,iu,q,ql,qr);
    } else if (dir == X2DIR) {
      GetOneVariableX2(k,j,il,iu,q,ql,qr);
    } else if (dir == X3DIR) {
      GetOneVariableX3(k,j,il,iu,q,ql,qr);
    }
  }

  // Velocities (add advection component)
  if (dir == X1DIR) {
    q.InitWithShallowSlice(prim,4,IVX,1);
    GetOneVariableX1(k,j,il,iu,q,v_l,v_r);
  } else if (dir == X2DIR) {
    q.InitWithShallowSlice(prim,4,IVY,1);
    GetOneVariableX2(k,j,il,iu,q,v_l,v_r);
  } else if (dir == X3DIR) {
    q.InitWithShallowSlice(prim,4,IVZ,1);
    GetOneVariableX3(k,j,il,iu,q,v_l,v_r);
  }

  return;
}

void CRIntegrator::GetOneVariableX1(int const k, int const j,
                                    int const il, int const iu,
				                            AthenaArray<Real> const &q,
                                    AthenaArray<Real> &ql,
				                            AthenaArray<Real> &qr)
{
  auto pco = pmy_cr->pmy_block->pcoord;
  Real dql,dqr,dqc,dq2;

#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    Real dx1im2 = pco->dx1v(i-2);
    Real dx1im1 = pco->dx1v(i-1);
    Real dx1i   = pco->dx1v(i);
    Real dxfr = pco->x1f(i) - pco->x1v(i-1);
    Real dxfl = pco->x1v(i) - pco->x1f(i);
    Real cfm = dx1im1/dxfr;
    Real cbm = dx1im2/(pco->x1v(i-1) - pco->x1f(i-1));
    Real cfp = dx1i/(pco->x1f(i+1) - pco->x1v(i));
    Real cbp = dx1im1/dxfl;

    dql   = (q(k,j,i-1) - q(k,j,i-2))/dx1im2;
    dqc   = (q(k,j,i  ) - q(k,j,i-1))/dx1im1;
    dqr   = (q(k,j,i+1) - q(k,j,i  ))/dx1i;

    // compute ql_(i-1/2) using Mignone 2014's modified van-Leer limiter
    dq2 = dql*dqc;
    ql(i) = q(k,j,i-1);
    if (dq2 > TINY_NUMBER)
      ql(i) += dxfr*dq2*(cfm*dql+cbm*dqc)/(dql*dql+(cfm+cbm-2.0)*dq2+dqc*dqc);

    // compute qr_(i-1/2) using Mignone 2014's modified van-Leer limiter
    dq2 = dqc*dqr;
    qr(i) =  q(k,j,i);
    if (dq2 > TINY_NUMBER)
      qr(i) -= dxfl*dq2*(cfp*dqc+cbp*dqr)/(dqc*dqc+(cfp+cbp-2.0)*dq2+dqr*dqr);
  }
}

void CRIntegrator::GetOneVariableX2(int const k, int const j,
                                    int const il, int const iu,
                                    AthenaArray<Real> const &q,
                                    AthenaArray<Real> &ql,
                                    AthenaArray<Real> &qr)
{
  auto pco = pmy_cr->pmy_block->pcoord;
  Real dql,dqr,dqc,dq2;

  Real dx2jm2 = pco->dx2v(j-2);
  Real dx2jm1 = pco->dx2v(j-1);
  Real dx2j   = pco->dx2v(j);
  Real dxfr = pco->x2f(j) - pco->x2v(j-1);
  Real dxfl = pco->x2v(j) - pco->x2f(j);
  Real cfm = dx2jm1/dxfr;
  Real cbm = dx2jm2/(pco->x2v(j-1) - pco->x2f(j-1));
  Real cfp = dx2j/(pco->x2f(j+1) - pco->x2v(j));
  Real cbp = dx2jm1/dxfl;

#pragma omp simd
  for (int i=il; i<=iu; ++i){
    dql = (q(k,j-1,i) - q(k,j-2,i))/dx2jm2;
    dqc = (q(k,j  ,i) - q(k,j-1,i))/dx2jm1;
    dqr = (q(k,j+1,i) - q(k,j  ,i))/dx2j;

    // Apply monotonicity constraints, compute ql_(i-1/2)
    dq2 = dql*dqc;
    ql(i) = q(k,j-1,i);
    if (dq2 > TINY_NUMBER)
      ql(i) += dxfr*dq2*(cfm*dql+cbm*dqc)/(dql*dql+(cfm+cbm-2.0)*dq2+dqc*dqc);

    // Apply monotonicity constraints, compute qr_(i-1/2)
    dq2 = dqc*dqr;
    qr(i) = q(k,j,i);
    if (dq2 > TINY_NUMBER)
      qr(i) -= dxfl*dq2*(cfp*dqc+cbp*dqr)/(dqc*dqc+(cfp+cbp-2.0)*dq2+dqr*dqr);
  }

}

void CRIntegrator::GetOneVariableX3(int const k, int const j,
                                    int const il, int const iu,
				                            AthenaArray<Real> const &q,
                                    AthenaArray<Real> &ql,
				                            AthenaArray<Real> &qr)
{
  auto pco = pmy_cr->pmy_block->pcoord;
  Real dql,dqr,dqc,dq2;

  Real dx3km2 = pco->dx3v(k-2);
  Real dx3km1 = pco->dx3v(k-1);
  Real dx3k   = pco->dx3v(k);
  Real dxfr = pco->x3f(k) - pco->x3v(k-1);
  Real dxfl = pco->x3v(k) - pco->x3f(k);
  Real cfm = dx3km1/dxfr;
  Real cbm = dx3km2/(pco->x3v(k-1) - pco->x3f(k-1));
  Real cfp = dx3k/(pco->x3f(k+1) - pco->x3v(k));
  Real cbp = dx3km1/dxfl;

#pragma omp simd
  for (int i=il; i<=iu; ++i){
    dql = (q(k-1,j,i) - q(k-2,j,i))/dx3km2;
    dqc = (q(k  ,j,i) - q(k-1,j,i))/dx3km1;
    dqr = (q(k+1,j,i) - q(k  ,j,i))/dx3k;

    // Apply monotonicity constraints, compute ql_(i-1/2)
    dq2 = dql*dqc;
    ql(i) = q(k-1,j,i);
    if (dq2 > TINY_NUMBER)
      ql(i) += dxfr*dq2*(cfm*dql+cbm*dqc)/(dql*dql+(cfm+cbm-2.0)*dq2+dqc*dqc);

    // Apply monotonicity constraints, compute qr_(i-1/2)
    dq2 = dqc*dqr;
    qr(i) = q(k,j,i);
    if (dq2 > TINY_NUMBER)
      qr(i) -= dxfl*dq2*(cfp*dqc+cbp*dqr)/(dqc*dqc+(cfp+cbp-2.0)*dq2+dqr*dqr);
  }
}
