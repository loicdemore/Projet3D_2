import SimpleITK as sitk

def est_lin_transf(im_ref, im_mov):
    """
    Estimate linear (affine) transform to align `im_mov` to `im_ref`
    and return the transform parameters.
    """
    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings
    registration_method.SetMetricAsMeanSquares()  # Can use other metrics like Mattes Mutual Information
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)  # Use a small percentage of pixels for speed
    
    # Interpolation
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    
    # Transformation model
    initial_transform = sitk.CenteredTransformInitializer(im_ref, im_mov, sitk.AffineTransform(im_ref.GetDimension()), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # Perform registration
    final_transform = registration_method.Execute(im_ref, im_mov)
    
    return final_transform

def est_nl_transf(im_ref, im_mov):
    """
    Estimate non-linear (BSpline) transform to align `im_mov` to `im_ref`
    and return the transform parameters.
    """
    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    # Interpolation
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer settings
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=50, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=500)
    
    # Transformation model (BSpline)
    grid_physical_spacing = [50.0] * im_ref.GetDimension()  # Grid spacing in physical units
    image_spacing = im_ref.GetSpacing()
    image_size = im_ref.GetSize()
    grid_size = [int(image_size[i] * image_spacing[i] / grid_physical_spacing[i] + 0.5) for i in range(im_ref.GetDimension())]
    grid_size = [max(3, grid_size[i]) for i in range(len(grid_size))]  # Ensure grid size is at least 3x3
    transform_domain_mesh_size = [g - 1 for g in grid_size]
    initial_transform = sitk.BSplineTransformInitializer(im_ref, transform_domain_mesh_size)
    registration_method.SetInitialTransformAsBSpline(initial_transform, inPlace=False)
    
    # Perform registration
    final_transform = registration_method.Execute(im_ref, im_mov)
    
    return final_transform

def apply_lin_transf(im_mov, lin_xfm):
    """
    Apply given linear transform `lin_xfm` to `im_mov` and return the transformed image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_mov)  # Use the moving image as reference
    resampler.SetTransform(lin_xfm)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    transformed_image = resampler.Execute(im_mov)
    return transformed_image

def apply_nl_transf(im_mov, nl_xfm):
    """
    Apply given non-linear transform `nl_xfm` to `im_mov` and return the transformed image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_mov)
    resampler.SetTransform(nl_xfm)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    transformed_image = resampler.Execute(im_mov)
    return transformed_image
