include(CMakePrintHelpers)


function(is_target_exist trgt)
	if(TARGET ${trgt})
		message("Target ${trgt} exists.")
	else()
		message("Target ${trgt} does NOT exist.")
	endif()
endfunction()
