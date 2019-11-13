function [xk1, Pkk, xkNew, PkNew] = rkf(F,G,V,H,D,Q,R,xk,Pk,uk,yk, lambda)
    % Robust update of KF: Returns predicted state, predicted error cov, 
    % corrected state and corrected error cov.  
    %  Model 
    %    x(k+1) = Fx(k) + Gu(k) + Vv(k)
    %    y(k) = Hx(k) + Du(k) + e(k) + w(k)
    %    v ~ N(0, Q)
    %    e ~ N(0, R)
    %    w sparse vector of outliers
   
    n = length(xk); 

    % Prediction
    xk1 = F*xk + G*uk;
    
    % Prediction covariance
    Pkk = F*Pk*F' + V*Q*V';

    
    % Correction
    if ~any(isnan(yk))
        % Innovations
        ytilde = yk - H*xk1 - D*uk;
        ZP = sqrtm(inv(Pk));
        ZR = sqrtm(inv(R));
        
        % Minimize using cvx 
        b = cat(1, ZR*ytilde, zeros(n,1));
        A = cat(1, ZR*H, ZP);
        cvx_begin quiet
            variable xtilde(n)
            minimize( sum( huber( A*xtilde-b , lambda) ) )
        cvx_end

        % Filter update
        xkNew = xk1 + xtilde;
        
        % Innovation cov
        S = H*Pkk*H' + R;
        
        % Kalman gain
        K = Pkk*H'*inv(S);      
        
        
        PkNew = (eye(n) - K*H)*Pkk;
    else
        xkNew = xk1;
        PkNew = Pkk;
    end
    % Returns xkNew, PkNew, xk1, Pkk,z
