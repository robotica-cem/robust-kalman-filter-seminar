function [xk1, Pkk, xkNew, PkNew, K] = kf(F,G,V,H,D,Q,R,xk,Pk,uk,yk)
    % Robust update of KF: Returns the corrected state and corrected error cov
    % and predicted state, predicted error cov, 
    % .
    %  Model 
    %    x(k+1) = Fx(k) + Gu(k) + Vv(k)
    %    y(k) = Hx(k) + Du(k) + e(k) 
    %    v ~ N(0, Q)
    %    e ~ N(0, R)
   
   
    % Prediction
    xk1 = F*xk + G*uk;
    n = length(xk1);
    
    % Prediction covariance
    Pkk = F*Pk*F' + V*Q*V';
    
    % Correction
    if ~any(isnan(yk))
        m = length(yk);
        % Innovations
        ek = yk - H*xk1 - D*uk;
        
        % Kalman gain
        S = H*Pkk*H' + R;
        K = Pkk*H'*inv(S);      
        
        % Filter update
        xkNew = xk1 + K*ek;

        PkNew = Pkk - K*H*Pkk;
 
        
        % Minimize using cvx 
        cvx_begin quiet
            variable x(n)
            minimize( matrix_frac(yk-D*uk-H*x, R) + matrix_frac(x-xk1, Pkk)) 
        cvx_end

        if (norm(x-xkNew) > 1e-4)
            sprintf('Ouch, not what I expected, norm(x-xkNew) = %f', norm(x-xkNew))
            x
            xkNew
        end
    else
        xkNew = xk1;
        PkNew = Pkk;
    end
    % Returns xkNew, PkNew, xk1, Pkk
