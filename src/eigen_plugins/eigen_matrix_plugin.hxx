
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix(eigen_zero_initialize_t)
    {
      Base::_check_template_params();
      Base::setZero(Base::SizeAtCompileTime);
    }

  
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix(eigen_ones_initialize_t)
    {
      Base::_check_template_params();
      Base::setOnes(Base::SizeAtCompileTime);
    }