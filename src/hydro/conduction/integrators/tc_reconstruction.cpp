// Athena++ headers
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../mesh/mesh.hpp"
#include "tc_integrators.hpp"
#include "../../../coordinates/coordinates.hpp"
#include "../tc.hpp"
#include "../../hydro.hpp"

void TCIntegrator::DonorCellX1(const int k, const int j,
  const int il, const int iu, const AthenaArray<Real> &u_tc,
  AthenaArray<Real> &rho, AthenaArray<Real> &tgas,
  AthenaArray<Real> &rho_l, AthenaArray<Real> &rho_r,
  AthenaArray<Real> &t_l, AthenaArray<Real> &t_r,
  AthenaArray<Real> &w_l, AthenaArray<Real> &w_r)
{

  for (int n=0; n<4; ++n){
#pragma simd
    for (int i=il; i<=iu; ++i){
      w_l(n,i) = u_tc(n,k,j,i-1);
      w_r(n,i) = u_tc(n,k,j,i  );
    }
  }

// gas temperature reconstruction
  for (int i=il; i<=iu; ++i){
    t_l(i) = tgas(k,j,i-1);
    t_r(i) = tgas(k,j,i  );
  }

// density reconstruction
  for (int i=il; i<=iu; ++i){
    rho_l(i) = rho(k,j,i-1);
    rho_r(i) = rho(k,j,i  );
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX2()
//  \brief

void TCIntegrator::DonorCellX2(const int k, const int j,
  const int il, const int iu, const AthenaArray<Real> &u_tc,
  AthenaArray<Real> &rho, AthenaArray<Real> &tgas,
  AthenaArray<Real> &rho_l, AthenaArray<Real> &rho_r,
  AthenaArray<Real> &t_l, AthenaArray<Real> &t_r,
  AthenaArray<Real> &w_l, AthenaArray<Real> &w_r)
{

  for (int n=0; n<4; ++n){
#pragma simd
    for (int i=il; i<=iu; ++i){
      w_l(n,i) = u_tc(n,k,j-1,i);
      w_r(n,i) = u_tc(n,k,j  ,i);
    }
  }

// gas temperature reconstruction
  for (int i=il; i<=iu; ++i){
    t_l(i) = tgas(k,j-1,i);
    t_r(i) = tgas(k,j,i  );
  }

// density reconstruction
  for (int i=il; i<=iu; ++i){
    rho_l(i) = rho(k,j-1,i);
    rho_r(i) = rho(k,j,i  );
  }


  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX3()
//  \brief

void TCIntegrator::DonorCellX3(const int k, const int j,
  const int il, const int iu, const AthenaArray<Real> &u_tc,
  AthenaArray<Real> &rho, AthenaArray<Real> &tgas,
  AthenaArray<Real> &rho_l, AthenaArray<Real> &rho_r,
  AthenaArray<Real> &t_l, AthenaArray<Real> &t_r,
  AthenaArray<Real> &w_l, AthenaArray<Real> &w_r)
{

  for (int n=0; n<4; ++n){
#pragma simd
    for (int i=il; i<=iu; ++i){
      w_l(n,i) = u_tc(n,k-1,j,i);
      w_r(n,i) = u_tc(n,k,j  ,i);
    }
  }

// gas temperature reconstruction
  for (int i=il; i<=iu; ++i){
    t_l(i) = tgas(k-1,j,i);
    t_r(i) = tgas(k,j,i  );
  }

// density reconstruction
  for (int i=il; i<=iu; ++i){
    rho_l(i) = rho(k-1,j,i);
    rho_r(i) = rho(k,j,i  );
  }


  return;
}



void TCIntegrator::PieceWiseLinear(const int k, const int j,
  const int il, const int iu, AthenaArray<Real> &u_tc,
  AthenaArray<Real> &rho, AthenaArray<Real> &tgas,
  AthenaArray<Real> &rho_l, AthenaArray<Real> &rho_r,
  AthenaArray<Real> &t_l, AthenaArray<Real> &t_r,
  AthenaArray<Real> &w_l, AthenaArray<Real> &w_r, int dir)
{

  AthenaArray<Real> q, ql, qr;

  for (int n=0; n<4; ++n) {
      q.InitWithShallowSlice(u_tc,4,n,1);
      ql.InitWithShallowSlice(w_l,2,n,1);
      qr.InitWithShallowSlice(w_r,2,n,1);
      switch(dir){
        case 0:
          GetOneVariableX1(k,j,il,iu,q,ql,qr);
          break;
        case 1:
          GetOneVariableX2(k,j,il,iu,q,ql,qr);
          break;
        case 2:
          GetOneVariableX3(k,j,il,iu,q,ql,qr);
          break;
        default:
          std::stringstream msg;
          msg << "### FATAL ERROR in NTC constructor" << std::endl
          << "Direction=" << dir << " not valid" << std::endl;
          throw std::runtime_error(msg.str().c_str());
      }
  }

//temperature , density reconstruction

  // add the advection component
  if(dir == 0){
    GetOneVariableX1(k,j,il,iu,tgas,t_l,t_r);
    GetOneVariableX1(k,j,il,iu,rho,rho_l,rho_r);
  }else if(dir==1){
    GetOneVariableX2(k,j,il,iu,tgas,t_l,t_r);
    GetOneVariableX2(k,j,il,iu,rho,rho_l,rho_r);
  }else if(dir==2){
    GetOneVariableX3(k,j,il,iu,tgas,t_l,t_r);
    GetOneVariableX3(k,j,il,iu,rho,rho_l,rho_r);
  }

  return;
}


void TCIntegrator::GetOneVariableX1(const int k, const int j,
      const int il, const int iu, const AthenaArray<Real> &q,
      AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{


  Coordinates *pco = pmy_tc->pmy_hydro->pmy_block->pcoord;
  Real dql,dqr,dqc,q_im1,q_i;

#pragma simd
  for (int i=il; i<=iu; ++i){
    Real& dx_im2 = pco->dx1v(i-2);
    Real& dx_im1 = pco->dx1v(i-1);
    Real& dx_i   = pco->dx1v(i);

    q_im1 = q(k,j,i-1);
    q_i   = q(k,j,i  );
    dql = (q(k,j,i-1) - q(k,j,i-2))/dx_im2;
    dqc = (q(k,j,i  ) - q(k,j,i-1))/dx_im1;
    dqr = (q(k,j,i+1) - q(k,j,i  ))/dx_i;
    // compute ql_(i-1/2) using Mignone 2014's modified van-Leer limiter
    Real dq2 = dql*dqc;
    ql(i) = q_im1;
    if(dq2>TINY_NUMBER) {
      Real dxfr=pco->x1f(i)-pco->x1v(i-1);
      Real cf=dx_im1/dxfr;
      Real cb=dx_im2/(pco->x1v(i-1)-pco->x1f(i-1));
      ql(i) += dxfr*dq2*(cf*dql+cb*dqc)/(dql*dql+(cf+cb-2.0)*dq2+dqc*dqc);
    }

    // compute qr_(i-1/2) using Mignone 2014's modified van-Leer limiter
    dq2 = dqc*dqr;
    qr(i) = q_i;
    if(dq2>TINY_NUMBER) {
      Real dxfl=pco->x1v(i)-pco->x1f(i);
      Real cf=dx_i/(pco->x1f(i+1)-pco->x1v(i));
      Real cb=dx_im1/dxfl;
      qr(i) -= dxfl*dq2*(cf*dqc+cb*dqr)/(dqc*dqc+(cf+cb-2.0)*dq2+dqr*dqr);
    }
  }
}

void TCIntegrator::GetOneVariableX2(const int k, const int j,
      const int il, const int iu, const AthenaArray<Real> &q,
      AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{


  Coordinates *pco = pmy_tc->pmy_hydro->pmy_block->pcoord;
  Real dx2jm2i = 1.0/pco->dx2v(j-2);
  Real dx2jm1i = 1.0/pco->dx2v(j-1);
  Real dx2ji   = 1.0/pco->dx2v(j);
  Real dxfr=pco->x2f(j)-pco->x2v(j-1);
  Real dxfl=pco->x2v(j)-pco->x2f(j);
  Real cfm=pco->dx2v(j-1)/dxfr;
  Real cbm=pco->dx2v(j-2)/(pco->x2v(j-1)-pco->x2f(j-1));
  Real cfp=pco->dx2v(j)/(pco->x2f(j+1)-pco->x2v(j));
  Real cbp=pco->dx2v(j-1)/dxfl;
  Real dql,dqr,dqc,q_jm1,q_j;

#pragma simd
  for (int i=il; i<=iu; ++i){
    q_jm1 = q(k,j-1,i);
    q_j   = q(k,j  ,i);
    dql = (q(k,j-1,i) - q(k,j-2,i))*dx2jm2i;
    dqc = (q(k,j  ,i) - q(k,j-1,i))*dx2jm1i;
    dqr = (q(k,j+1,i) - q(k,j  ,i))*dx2ji;

    // Apply monotonicity constraints, compute ql_(i-1/2)
    Real dq2 = dql*dqc;
    ql(i) = q_jm1;
    if(dq2>TINY_NUMBER)
      ql(i) += dxfr*dq2*(cfm*dql+cbm*dqc)/(dql*dql+(cfm+cbm-2.0)*dq2+dqc*dqc);

    // Apply monotonicity constraints, compute qr_(i-1/2)
    dq2 = dqc*dqr;
    qr(i) = q_j;
    if(dq2>TINY_NUMBER)
      qr(i) -= dxfl*dq2*(cfp*dqc+cbp*dqr)/(dqc*dqc+(cfp+cbp-2.0)*dq2+dqr*dqr);
  }

}

void TCIntegrator::GetOneVariableX3(const int k, const int j,
      const int il, const int iu, const AthenaArray<Real> &q,
      AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{


  Coordinates *pco = pmy_tc->pmy_hydro->pmy_block->pcoord;
  Real dx3km2i = 1.0/pco->dx3v(k-2);
  Real dx3km1i = 1.0/pco->dx3v(k-1);
  Real dx3ki   = 1.0/pco->dx3v(k);
  Real dxfr=pco->x3f(k)-pco->x3v(k-1);
  Real dxfl=pco->x3v(k)-pco->x3f(k);
  Real cfm=pco->dx3v(k-1)/dxfr;
  Real cbm=pco->dx3v(k-2)/(pco->x3v(k-1)-pco->x3f(k-1));
  Real cfp=pco->dx3v(k)/(pco->x3f(k+1)-pco->x3v(k));
  Real cbp=pco->dx3v(k-1)/dxfl;
  Real dql,dqr,dqc,q_km1,q_k;

#pragma simd
  for (int i=il; i<=iu; ++i){
    q_km1 = q(k-1,j,i);
    q_k   = q(k  ,j,i);
    dql = (q(k-1,j,i) - q(k-2,j,i))*dx3km2i;
    dqc = (q(k  ,j,i) - q(k-1,j,i))*dx3km1i;
    dqr = (q(k+1,j,i) - q(k  ,j,i))*dx3ki;

    // Apply monotonicity constraints, compute ql_(i-1/2)
    Real dq2 = dql*dqc;
    ql(i) = q_km1;
    if(dq2>TINY_NUMBER)
      ql(i) += dxfr*dq2*(cfm*dql+cbm*dqc)/(dql*dql+(cfm+cbm-2.0)*dq2+dqc*dqc);

    // Apply monotonicity constraints, compute qr_(i-1/2)
    dq2 = dqc*dqr;
    qr(i) = q_k;
    if(dq2>TINY_NUMBER)
      qr(i) -= dxfl*dq2*(cfp*dqc+cbp*dqr)/(dqc*dqc+(cfp+cbp-2.0)*dq2+dqr*dqr);
  }
}
