
#parameters

set(OpenCV3_CMAKE_SCRIPTS "${OpenCV3_DIR}/share/OpenCV")
set(OpenCV3_INCLUDE_DIRS "${OpenCV3_DIR}/include")
set(OpenCV3_LIB_DIR "${OpenCV3_DIR}/lib")
set(OpenCV3_VERSION 3.1.0)

#library 


IF(OSX)
  set(OpenCV3_LIBS
	${OpenCV3_LIBS}
	${OpenCV3_LIB_DIR}/libopencv_viz.dylib
	${OpenCV3_LIB_DIR}/libopencv_videostab.dylib
	${OpenCV3_LIB_DIR}/libopencv_videoio.dylib
	${OpenCV3_LIB_DIR}/libopencv_video.dylib
	${OpenCV3_LIB_DIR}/libopencv_superres.dylib
	${OpenCV3_LIB_DIR}/libopencv_stitching.dylib
	${OpenCV3_LIB_DIR}/libopencv_shape.dylib
	${OpenCV3_LIB_DIR}/libopencv_photo.dylib
	${OpenCV3_LIB_DIR}/libopencv_objdetect.dylib
	${OpenCV3_LIB_DIR}/libopencv_ml.dylib
	${OpenCV3_LIB_DIR}/libopencv_imgproc.dylib
	${OpenCV3_LIB_DIR}/libopencv_imgcodecs.dylib
	${OpenCV3_LIB_DIR}/libopencv_highgui.dylib
	${OpenCV3_LIB_DIR}/libopencv_flann.dylib
	${OpenCV3_LIB_DIR}/libopencv_features2d.dylib
	${OpenCV3_LIB_DIR}/libopencv_core.dylib
	${OpenCV3_LIB_DIR}/libopencv_calib3d.dylib 
	)
ELSE(OSX)
set(OpenCV3_LIBS
	${OpenCV3_LIBS}
	${OpenCV3_LIB_DIR}/libopencv_viz.so
	${OpenCV3_LIB_DIR}/libopencv_videostab.so
	${OpenCV3_LIB_DIR}/libopencv_videoio.so
	${OpenCV3_LIB_DIR}/libopencv_video.so
	${OpenCV3_LIB_DIR}/libopencv_superres.so
	${OpenCV3_LIB_DIR}/libopencv_stitching.so
	${OpenCV3_LIB_DIR}/libopencv_shape.so 
	${OpenCV3_LIB_DIR}/libopencv_photo.so
	${OpenCV3_LIB_DIR}/libopencv_objdetect.so
	${OpenCV3_LIB_DIR}/libopencv_ml.so
	${OpenCV3_LIB_DIR}/libopencv_imgproc.so
	${OpenCV3_LIB_DIR}/libopencv_imgcodecs.so
	${OpenCV3_LIB_DIR}/libopencv_highgui.so
	${OpenCV3_LIB_DIR}/libopencv_flann.so
	${OpenCV3_LIB_DIR}/libopencv_features2d.so
	${OpenCV3_LIB_DIR}/libopencv_core.so
	${OpenCV3_LIB_DIR}/libopencv_calib3d.so 
	)
ENDIF(OSX)


	
#unset(OpenCV3_INCLUDE_DIRS)
include(FindPackageHandleStandardArgs)


find_package_handle_standard_args(
	OpenCV3 
	FOUND_VAR OpenCV3_FOUND
    REQUIRED_VARS OpenCV3_DIR OpenCV3_LIB_DIR OpenCV3_LIBS OpenCV3_INCLUDE_DIRS
    VERSION_VAR OpenCV3_VERSION_STRING
)
#Check at least if the core library of opencv is there.

IF(OSX)
 if(EXISTS ${OpenCV3_LIB_DIR}/libopencv_core.dylib)
   set(OpenCV3_FOUND "TRUE")
 else(EXISTS ${OpenCV3_LIB_DIR}/libopencv_core.dylib)
   set(OpenCV3_FOUND "OpenCV3_FOUND-NOTFOUND")
 endif(EXISTS ${OpenCV3_LIB_DIR}/libopencv_core.dylib)
ELSE(OSX)
 if(EXISTS ${OpenCV3_LIB_DIR}/libopencv_core.so)
   set(OpenCV3_FOUND "TRUE")
 else(EXISTS ${OpenCV3_LIB_DIR}/libopencv_core.so)
   set(OpenCV3_FOUND "OpenCV3_FOUND-NOTFOUND")
 endif(EXISTS ${OpenCV3_LIB_DIR}/libopencv_core.so)
ENDIF(OSX)


