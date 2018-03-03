function [phi] = return_phi(arg,psize)
    if arg == 1
        phi = kron(dctmtx(psize)',dctmtx(psize)');
    elseif arg == 2
        phi = normrnd(0,1.0,32,64);
    end
end