function [xk1, Pkk, xkNew, PkNew, z] = rkf2(H,G,F,C,D,Q,R,xk,Pk,uk,yk, lambda)
    % Robust update of KF using explicit solution to l1-regularization
    % problem for 2d measurements.  
    % Returns predicted state, predicted error cov, 
    % corrected state and corrected error cov. Will also return the vector
    % z obtained from the l1-regularization problem. 
    %  Model 
    %    x(k+1) = Hx(k) + Gu(k) + Fv(k)
    %    y(k) = Cx(k) + Du(k) + e(k) + w(k)
    %    v ~ N(0, Q)
    %    e ~ N(0, R)
    %    w sparse vector of outliers
   
    m = length(yk); 

    if m ~= 2
        error('This robust kalman filter works only for length(y)=2')
    end
    
    % Prediction
    xk1 = H*xk + G*uk;
    
    % Prediction covariance
    Pkk = H*Pk*H' + F*Q*F';

    
    % Correction
    if ~any(isnan(yk))
        % Innovations
        ek = yk - C*xk1 - D*uk;
        
        % Kalman gain
        CPCR = C*Pkk*C' + R;
        K = Pkk*C'/CPCR;      
        
        % Compute weighting matrix to be used in minimization problem
        ICK = (eye(m)-C*K);
        S = ICK' / R * ICK + K' / Pkk * K;
        
        % Minimize f = 0.5*(e-z)'*W*(e-z) + lambda*1*z
        % assuming zi to be same sign as ei
        % Works only if S is diagonal, so lets force it
        % We will need the inverse only
        Sinv = diag(1.0./diag(S));
        se = sign(ek);
        z = ek - lambda*Sinv*se;
        z(find(sign(z) ~= se)) = 0;
        
        % Filter update
        xkNew = xk1 + K*(ek - z);

        PkNew = Pkk - K*C*Pkk;
    else
        xkNew = xk1;
        PkNew = Pkk;
    end
    % Returns xkNew, PkNew, xk1, Pkk,z
