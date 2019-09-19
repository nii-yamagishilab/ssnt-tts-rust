import os
import platform
import re
import setuptools
import sys
import unittest
from setuptools.command.build_ext import build_ext
import tensorflow as tf

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

ssnt_tts_path = "../target/release"

tf_include = tf.sysconfig.get_include()
tf_src_dir = tf.sysconfig.get_lib()
tf_includes = [tf_include, tf_src_dir]
include_dirs = tf_includes

lib_srcs = ['src/ssnt_tts_beam_search_decode_op.cc']

TF_CXX11_ABI = "0"
extra_compile_args = ['-std=c++11', '-fPIC', '-O2', '-D_GLIBCXX_USE_CXX11_ABI=' + TF_CXX11_ABI]
extra_compile_args += tf.sysconfig.get_compile_flags()
if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++']

ext = setuptools.Extension('ssnt_tts_tensorflow.kernels',
                           sources=lib_srcs,
                           language='c++',
                           include_dirs=include_dirs,
                           library_dirs=[ssnt_tts_path],
                           runtime_library_dirs=[os.path.realpath(ssnt_tts_path)],
                           libraries=['ssnt_tts_c'],
                           extra_compile_args=extra_compile_args,
                           extra_link_args=tf.sysconfig.get_link_flags())

def cmakelist(ext):
    definition = f'''
project(ssnt_tts_tensorflow CXX)
add_library(rnnt-tensorflow SHARED {" ".join(ext.sources)})
include_directories({" ".join(ext.include_dirs)})
add_definitions({" ".join(ext.define_macros)})
target_link_libraries(ssnt_tts_tensorflow "{ext.library_dirs}")
    '''
    return definition

print(cmakelist(ext))


class build_tf_ext(build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        build_ext.build_extensions(self)


def discover_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite


setuptools.setup(
    name="ssnt_tts_tensorflow",
    version="0.1",
    description="TensorFlow wrapper for rnn-transducer",
    url="https://github.com/TanUkkii007/",
    author="Yusuke Yasuda",
    packages=["ssnt_tts_tensorflow"],
    ext_modules=[ext],
    cmdclass={'build_ext': build_tf_ext},
    test_suite='setup.discover_test_suite',
)
