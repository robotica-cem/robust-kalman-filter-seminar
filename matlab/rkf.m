function [xk1, Pkk, xkNew, PkNew, z] = rkf(H,G,F,C,D,Q,R,xk,Pk,uk,yk, lambda)
    % Robust update of KF: Returns predicted state, predicted error cov, 
    % corrected state and corrected error cov. Will also return the vector
    % z obtained from the l1-regularization problem. 
    %  Model 
    %    x(k+1) = Hx(k) + Gu(k) + Fv(k)
    %    y(k) = Cx(k) + Du(k) + e(k) + w(k)
    %    v ~ N(0, Q)
    %    e ~ N(0, R)
    %    w sparse vector of outliers
   
    m = length(yk); 

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
        W = ICK' / R * ICK + K' / Pkk * K;
        
        % Minimize using cvx
        cvx_begin quiet
            variable z(m)
            minimize( 0.5*(ek-z)'*W*(ek-z) + lambda*norm(z, 1) )
        cvx_end
        
        % debug
        if norm(ek) > 1
            [ek z]
        end
        % Filter update
        xkNew = xk1 + K*(ek - z);

        PkNew = Pkk - K*C*Pkk;
    else
        xkNew = xk1;
        PkNew = Pkk;
    end
    % Returns xkNew, PkNew, xk1, Pkk,z
