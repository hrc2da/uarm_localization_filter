function [xhat_u, P_u, rej] = HIRO_KF(xk, uk, P_k, z_kp1, F, B, G, H, Q, R, Lam0)
    %kalman filter funciotn for HIRO project
   
    %predict
    xhat_p = F*xk + B*uk;
    P_p = F*P_k*F' + G*Q*G';
    
    %Innovations calculations
    inn=(z_kp1 - H*xhat_p); %innovations
    S = H*P_p*H' + R; %innovations covariance
    
    %Kalman Gain
    K = P_p*H'*inv(S);
    
    %reject Measurements
    Lam=inn'*inv(S)*inn;
    if Lam>Lam0
        % no update (use predictions)
        xhat_u = xhat_p;
        P_u = P_p;
        rej = 1;
    else
        % Update
        xhat_u = xhat_p + K*inn;
        P_u = (eye(3)-K*H)*P_p*(eye(3)-K*H)'+K*R*K';
        rej = 0;
    end
end