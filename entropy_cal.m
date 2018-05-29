function window_entropy = entropy_cal(win_accel_data)

    %check if the window has negative valued data points
    window_size = rows(win_accel_data);
    max_negative_xval = 0;
    max_negative_yval = 0;
    max_negative_zval = 0;
    for i = 1 : window_size
       if (win_accel_data(i,1) < max_negative_xval)
          max_negative_xval = win_accel_data(i,1);
       end
       if (win_accel_data(i,2) < max_negative_yval)
          max_negative_yval = win_accel_data(i,2);
       end
       if (win_accel_data(i,3) < max_negative_zval)
          max_negative_zval = win_accel_data(i,3);
       end
    end
    
    %disp(max_negative_xval);
    
    %make the accel data non-negative
    if (max_negative_xval < 0)
      for i = 1 : window_size
       win_accel_data(i,1) = 1 + (win_accel_data(i,1) + (-max_negative_xval));
      end
    end
    
    
    %disp(win_accel_data);
    
    if (max_negative_yval < 0)
      for i = 1 : window_size
       win_accel_data(i,2) = 1 + (win_accel_data(i,2) + (-max_negative_yval));
      end
    end
    
    if (max_negative_zval < 0)
      for i = 1 : window_size
       win_accel_data(i,3) = 1 + (win_accel_data(i,3) + (-max_negative_zval));
      end
    end
    
    
    %find the constant k for to convert accel data to a probability distribution
    %function (pdf)
    
    kx = sum(win_accel_data(:,1));
    ky = sum(win_accel_data(:,2));
    kz = sum(win_accel_data(:,3));
    for i = 1 : window_size
      win_accel_data(i,1) = win_accel_data(i,1)/kx;
      win_accel_data(i,2) = win_accel_data(i,2)/ky;
      win_accel_data(i,3) = win_accel_data(i,3)/kz;
    end
    
    %disp(kx);
    
    
    x_window_entropy = 0;
    y_window_entropy = 0;
    z_window_entropy = 0;
    for i = 1 : window_size
      x_window_entropy = x_window_entropy + win_accel_data(i,1)*log2(win_accel_data(i,1));
      y_window_entropy = y_window_entropy + win_accel_data(i,2)*log2(win_accel_data(i,2));
      z_window_entropy = z_window_entropy + win_accel_data(i,3)*log2(win_accel_data(i,3));
    end
    
    window_entropy = zeros(1,3);
    window_entropy(1,1) = -x_window_entropy;
    window_entropy(1,2) = -y_window_entropy;
    window_entropy(1,3) = -z_window_entropy;
    
 endfunction
    

