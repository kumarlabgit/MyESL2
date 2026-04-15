#pragma once

// Port of /claude/MyESL/src/epph.h — L1/Lq Euclidean projection primitives.
// Functions are marked static inline so this header can be included in multiple
// translation units without ODR violations. Semantic behavior is unchanged
// from the MyESL source.

#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace gl_logr_epph {

#ifndef GL_LOGR_EPPH_DELTA
#define GL_LOGR_EPPH_DELTA 1e-8
#endif
#ifndef GL_LOGR_EPPH_INNERITER
#define GL_LOGR_EPPH_INNERITER 1000
#endif
#ifndef GL_LOGR_EPPH_OUTERITER
#define GL_LOGR_EPPH_OUTERITER 1000
#endif

// Local aliases mirroring the MyESL source's bare identifiers.
static constexpr double delta = GL_LOGR_EPPH_DELTA;
static constexpr int innerIter = GL_LOGR_EPPH_INNERITER;
static constexpr int outerIter = GL_LOGR_EPPH_OUTERITER;

/*
-------------------------- Function eplb -----------------------------

 Euclidean Projection onto l1 Ball (eplb)

        min  1/2 ||x- v||_2^2
        s.t. ||x||_1 <= z
 */
static inline void eplb(double * x, double *root, int * steps, double * v, int n, double z, double lambda0)
{
    int i, j, flag=0;
    int rho_1, rho_2, rho, rho_T, rho_S;
    int V_i_b, V_i_e, V_i;
    double lambda_1, lambda_2, lambda_T, lambda_S, lambda;
    double s_1, s_2, s, s_T, s_S, v_max, temp;
    double f_lambda_1, f_lambda_2, f_lambda, f_lambda_T, f_lambda_S;
    int iter_step=0;

    if (z< 0){
        printf("\n z should be nonnegative!");
        return;
    }

    V_i=0;
    if (v[0] !=0){
        rho_1=1;
        s_1=x[V_i]=v_max=std::fabs(v[0]);
        V_i++;
    }
    else{
        rho_1=0;
        s_1=v_max=0;
    }

    for (i=1;i<n; i++){
        if (v[i]!=0){
            x[V_i]=std::fabs(v[i]); s_1+= x[V_i]; rho_1++;

            if (x[V_i] > v_max)
                v_max=x[V_i];
            V_i++;
        }
    }

    if (s_1 <= z){
        flag=1;        lambda=0;
        for(i=0;i<n;i++){
            x[i]=v[i];
        }
        *root=lambda;
        *steps=iter_step;
        return;
    }

    lambda_1=0; lambda_2=v_max;
    f_lambda_1=s_1 -z;
    rho_2=0; s_2=0; f_lambda_2=-z;
    V_i_b=0; V_i_e=V_i-1;

    lambda=lambda0;
    if ( (lambda<lambda_2) && (lambda> lambda_1) ){
        i=V_i_b; j=V_i_e; rho=0; s=0;
        while (i <= j){
            while( (i <= V_i_e) && (x[i] <= lambda) ){
                i++;
            }
            while( (j>=V_i_b) && (x[j] > lambda) ){
                s+=x[j];
                j--;
            }
            if (i<j){
                s+=x[i];

                temp=x[i];  x[i]=x[j];  x[j]=temp;
                i++;  j--;
            }
        }

        rho=V_i_e-j;  rho+=rho_2;  s+=s_2;
        f_lambda=s-rho*lambda-z;

        if ( std::fabs(f_lambda)< delta ){
            flag=1;
        }

        if (f_lambda <0){
            lambda_2=lambda; s_2=s; rho_2=rho; f_lambda_2=f_lambda;

            V_i_e=j;  V_i=V_i_e-V_i_b+1;
        }
        else{
            lambda_1=lambda; rho_1=rho; s_1=s; f_lambda_1=f_lambda;

            V_i_b=i; V_i=V_i_e-V_i_b+1;
        }

        if (V_i==0){
            lambda=(s - z)/ rho;
            flag=1;
        }
    }

    while (!flag){
        iter_step++;

        lambda_T=lambda_1 + f_lambda_1 /rho_1;
        if(rho_2 !=0){
            if (lambda_2 + f_lambda_2 /rho_2 >    lambda_T)
                lambda_T=lambda_2 + f_lambda_2 /rho_2;
        }

        lambda_S=lambda_2 - f_lambda_2 *(lambda_2-lambda_1)/(f_lambda_2-f_lambda_1);

        if (std::fabs(lambda_T-lambda_S) <= delta){
            lambda=lambda_T; flag=1;
            break;
        }

        lambda=(lambda_T+lambda_S)/2;

        s_T=s_S=s=0;
        rho_T=rho_S=rho=0;
        i=V_i_b; j=V_i_e;
        while (i <= j){
            while( (i <= V_i_e) && (x[i] <= lambda) ){
                if (x[i]> lambda_T){
                    s_T+=x[i]; rho_T++;
                }
                i++;
            }
            while( (j>=V_i_b) && (x[j] > lambda) ){
                if (x[j] > lambda_S){
                    s_S+=x[j]; rho_S++;
                }
                else{
                    s+=x[j];  rho++;
                }
                j--;
            }
            if (i<j){
                if (x[i] > lambda_S){
                    s_S+=x[i]; rho_S++;
                }
                else{
                    s+=x[i]; rho++;
                }

                if (x[j]> lambda_T){
                    s_T+=x[j]; rho_T++;
                }

                temp=x[i]; x[i]=x[j];  x[j]=temp;
                i++; j--;
            }
        }

        s_S+=s_2; rho_S+=rho_2;
        s+=s_S; rho+=rho_S;
        s_T+=s; rho_T+=rho;
        f_lambda_S=s_S-rho_S*lambda_S-z;
        f_lambda=s-rho*lambda-z;
        f_lambda_T=s_T-rho_T*lambda_T-z;

        if ( std::fabs(f_lambda)< delta ){
            flag=1;
            break;
        }
        if ( std::fabs(f_lambda_S)< delta ){
            lambda=lambda_S; flag=1;
            break;
        }
        if ( std::fabs(f_lambda_T)< delta ){
            lambda=lambda_T; flag=1;
            break;
        }

        if (f_lambda <0){
            lambda_2=lambda;  s_2=s;  rho_2=rho;
            f_lambda_2=f_lambda;

            lambda_1=lambda_T; s_1=s_T; rho_1=rho_T;
            f_lambda_1=f_lambda_T;

            V_i_e=j;  i=V_i_b;
            while (i <= j){
                while( (i <= V_i_e) && (x[i] <= lambda_T) ){
                    i++;
                }
                while( (j>=V_i_b) && (x[j] > lambda_T) ){
                    j--;
                }
                if (i<j){
                    x[j]=x[i];
                    i++;   j--;
                }
            }
            V_i_b=i; V_i=V_i_e-V_i_b+1;
        }
        else{
            lambda_1=lambda;  s_1=s; rho_1=rho;
            f_lambda_1=f_lambda;

            lambda_2=lambda_S; s_2=s_S; rho_2=rho_S;
            f_lambda_2=f_lambda_S;

            V_i_b=i;  j=V_i_e;
            while (i <= j){
                while( (i <= V_i_e) && (x[i] <= lambda_S) ){
                    i++;
                }
                while( (j>=V_i_b) && (x[j] > lambda_S) ){
                    j--;
                }
                if (i<j){
                    x[i]=x[j];
                    i++;   j--;
                }
            }
            V_i_e=j; V_i=V_i_e-V_i_b+1;
        }

        if (V_i==0){
            lambda=(s - z)/ rho; flag=1;
            break;
        }
    }


    for(i=0;i<n;i++){
        if (v[i] > lambda)
            x[i]=v[i]-lambda;
        else
            if (v[i]< -lambda)
                x[i]=v[i]+lambda;
            else
                x[i]=0;
    }
    *root=lambda;
    *steps=iter_step;
}

/*
-------------------------- Function epp1 -----------------------------

 The L1-norm Regularized Euclidean Projection (epp1)

        min  1/2 ||x- v||_2^2 + rho ||x||_1
 */
static inline void epp1(double *x, double *v, int n, double rho){
    int i;
    for(i=0;i<n;i++){
        if (std::fabs(v[i])<=rho)
            x[i]=0;
        else
            if (v[i]< -rho)
                x[i]=v[i]+rho;
            else
                x[i]=v[i]-rho;
    }
}

/*
-------------------------- Function epp2 -----------------------------

 The L2-norm Regularized Euclidean Projection (epp2)

        min  1/2 ||x- v||_2^2 + rho ||x||_2
 */
static inline void epp2(double *x, double *v, int n, double rho){
    int i;
    double v2=0, ratio;

    for(i=0; i< n; i++){
        v2+=v[i]*v[i];
    }
    v2=std::sqrt(v2);

    if (rho >= v2)
        for(i=0;i<n;i++)
            x[i]=0;
    else{
        ratio= (v2-rho) /v2;
        for(i=0;i<n;i++)
            x[i]=v[i]*ratio;
    }
}

/*
-------------------------- Function eppInf -----------------------------

 The LInf-norm Regularized Euclidean Projection (eppInf)
 */
static inline void eppInf(double *x, double * c, int * iter_step, double *v, int n, double rho, double c0){
    int i, steps;

    eplb(x, c, &steps, v, n, rho, c0);

    for(i=0; i< n; i++){
        x[i]=v[i]-x[i];
    }
    iter_step[0]=steps;
    iter_step[1]=0;
}

/*
-------------------------- Function zerofind -----------------------------

 Find the root for the function: f(x) = x + c x^{p-1} - v,
 */
static inline void zerofind(double *root, int * iterStep, double v, double p, double c, double x0){

    double x, f, fprime, p1=p-1, pp;
    int step=0;

    if (v==0){
        *root=0;       *iterStep=0;   return;
    }

    if (c==0){
        *root=v;       * iterStep=0;  return;
    }

    if ( (x0 <v) && (x0>0) )
        x=x0;
    else
        x=v;


    pp=std::pow(x, p1);
    f= x + c* pp -v;

    while (1){
        step++;

        fprime=1 + c* p1 * pp / x;

        x = x- f/fprime;

        if (p>2){
            if (x>v){
                x=v;
            }
        }
        else{
            if ( (x<0) || (x>v)){
                x=1e-30;

                f= x+c* std::pow(x,p1)-v;

                if (f>0){
                    *root=x;
                    * iterStep=step;
                    break;
                }
            }
        }

        pp=std::pow(x, p1);
        f= x + c* pp -v;

        if ( std::fabs(f) <= delta){
            *root=x;
            * iterStep=step;
            break;
        }

        if (step>=innerIter){
            printf("\n The number of steps exceed %d, in finding the root for f(x)= x + c x^{p-1} - v, 0< x< v.", innerIter);
            printf("\n If you meet with this problem, please contact Jun Liu (j.liu@asu.edu). Thanks!");
            return;
        }
    }
}

/*
-------------------------- Function norm -----------------------------
 Compute the p-norm
*/
static inline double norm(double * v, double p, int n){
    int i;
    double t=0;

    for(i=0;i<n;i++)
        t+=std::pow(v[i], p);

    return( std::pow(t, 1/p) );
}

/*
-------------------------- Function eppO -----------------------------

 The Lp-norm Regularized Euclidean Projection (eppO) for 1< p<Inf
 */
static inline void eppO(double *x, double * cc, int * iter_step, double *v, int n, double rho, double p){

    int i, *flag, bisStep, newtonStep=0, totoalStep=0;
    double vq=0, epsilon, vmax=0, vmin=1e10;
    double q=1/(1-1/p), c, c1, c2, root, f, xp;

    double x_diff=0;
    double temp;
    int p_n=1;

    flag=(int *)std::malloc(sizeof(int)*n);

    for(i=0; i< n; i++){

        x[i]=0;

        if (v[i]==0)
            flag[i]=0;
        else
        {
            if (v[i]>0)
                flag[i]=0;
            else
            {
                flag[i]=1;
                v[i]=-v[i];
            }

            vq+=std::pow(v[i], q);


            if (v[i]>vmax)
                vmax=v[i];

            if (v[i]<vmin)
                vmin=v[i];
        }
    }
    vq=std::pow(vq, 1/q);

    if (rho >= vq){
        *cc=0;
        iter_step[0]=iter_step[1]=0;


        for(i=0;i<n;i++){
            if (flag[i])
                v[i]=-v[i];
        }

        std::free(flag);
        return;
    }

    epsilon=(vq -rho)/ vq;
    if (p>2){

        if ( std::log((1-epsilon) * vmin) - (p-1) * std::log( epsilon* vmin ) >= 709 )
        {
            for(i=0;i<n;i++){
                if (flag[i])
                    v[i]=-v[i];
            }

            eppInf(x, cc, iter_step, v,  n, rho, 0);

            std::free(flag);
            return;
        }

        c1= (1-epsilon) * vmax / std::pow(epsilon* vmax, p-1);
        c2= (1-epsilon) * vmin / std::pow(epsilon* vmin, p-1);
    }
    else{

        c2= (1-epsilon) * vmax / std::pow(epsilon* vmax, p-1);
        c1= (1-epsilon) * vmin / std::pow(epsilon* vmin, p-1);
    }

    if (std::fabs(c1-c2) <= delta){
        c=c1;
    }
    else
        c=(c1+c2)/2;


    bisStep =0;

    while(1){
        bisStep++;

        x_diff=0;
        for(i=0;i<n;i++){
            zerofind(&root, &newtonStep, v[i], p, c, x[i]);

            temp=std::fabs(root-x[i]);
            if (x_diff< temp )
                x_diff=temp;

            x[i]=root;
            totoalStep+=newtonStep;
        }

        xp=norm(x, p, n);

        f=rho * std::pow(xp, 1-p) - c;

        if ( std::fabs(f)<=delta || std::fabs(c1-c2)<=delta )
            break;
        else{
            if (f>0){
                if ( (x_diff <=delta) && (p_n==0) )
                    break;

                c1=c;  p_n=1;
            }
            else{

                if ( (x_diff <=delta) && (p_n==1) )
                    break;

                c2=c;  p_n=0;
            }
        }
        c=(c1+c2)/2;

        if (bisStep>=outerIter){


            if ( std::fabs(c1-c2) <=delta * c2 )
                break;
            else{
                printf("\n The number of bisection steps exceed %d.", outerIter);
                printf("\n c1=%e, c2=%e, x_diff=%e, f=%e",c1,c2,x_diff,f);
                printf("\n If you meet with this problem, please contact Jun Liu (j.liu@asu.edu). Thanks!");

                return;
            }
        }
    }

    for(i=0;i<n;i++){
        if (flag[i]){
            x[i]=-x[i];
            v[i]=-v[i];
        }
    }
    std::free(flag);

    *cc=c;

    iter_step[0]=bisStep;
    iter_step[1]=totoalStep;
}

/*
-------------------------- Function epp -----------------------------

 The Lp-norm Regularized Euclidean Projection (epp) for all p>=1
 */
static inline void epp(double *x, double * c, int * iter_step, double * v, int n, double rho, double p, double c0){

    if (rho <0){
        printf("\n rho should be non-negative!");
        std::exit(1);
    }

    if (p==1){
        epp1(x, v, n, rho);
        *c=0;
        iter_step[0]=iter_step[1]=0;
    }
    else
        if (p==2){
            epp2(x, v, n, rho);
            *c=0;
            iter_step[0]=iter_step[1]=0;
        }
        else
            if (p>=1e6)
                eppInf(x, c, iter_step, v,  n, rho, c0);
            else
                eppO(x, c, iter_step, v,  n, rho, p);
}

}  // namespace gl_logr_epph
