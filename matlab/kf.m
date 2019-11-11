function [xk1, Pkk, xkNew, PkNew, K] = kf(H,G,F,C,D,Q,R,xk,Pk,uk,yk)
    % Robust update of KF: Returns the corrected state and corrected error cov
    % and predicted state, predicted error cov, 
    % .
    %  Model 
    %    x(k+1) = Hx(k) + Gu(k) + Fv(k)
    %    y(k) = Cx(k) + Du(k) + e(k) 
    %    v ~ N(0, Q)
    %    e ~ N(0, R)
   
   
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
        K = Pkk*C'*inv(CPCR);      
        
        % Filter update
        xkNew = xk1 + K*ek;

        PkNew = Pkk - K*C*Pkk;
    else
        xkNew = xk1;
        PkNew = Pkk;
    end
    % Returns xkNew, PkNew, xk1, Pkk
