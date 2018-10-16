import ctypes

#load DLL into memory.
sarkdll0 = ctypes.WinDLL(
        'C:\\Users\\a6q\\Desktop\\SARK_Plots_Proto-23Feb2018\\EPPlus.dll')



sarkdll = ctypes.WinDLL(
        'C:\\Users\\a6q\\Desktop\\SARK_Plots_Proto-23Feb2018\\SARK110_DLL.dll')

#result= sarkdll.SARK_Version(5)
#print(result)




# Set up prototype and parameters for the desired function call.
# HLLAPI













'''
from ctypes import*

# give location of dll
sarkdll = cdll.LoadLibrary(
        'C:\\Users\\a6q\\Desktop\\SARK_Plots_Proto-23Feb2018\\SARK110_DLL.dll')

#result= sarkdll.Sark_Version(5)


#print(result)

'''
