# Copyright (c) 2006-2008 The Regents of The University of Michigan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
from os.path import basename, exists, join as joinpath, normpath
from os.path import isdir, isfile, islink

cpu2017_root = '/home/dilshad/Downloads/Project downloads/benchmark/cpu2017'

spec_dist = os.environ.get('M5_CPU2017', f'{cpu2017_root}')

def copyfiles(srcdir, dstdir):
    from filecmp import cmp as filecmp
    from shutil import copyfile

    srcdir = normpath(srcdir)
    dstdir = normpath(dstdir)

    if not isdir(dstdir):
        os.mkdir(dstdir)

    for root, dirs, files in os.walk(srcdir):
        root = normpath(root)
        prefix = os.path.commonprefix([root, srcdir])

        root = root[len(prefix):]
        if root.startswith('/'):
            root = root[1:]

        for entry in dirs:
            newdir = joinpath(dstdir, root, entry)
            if not isdir(newdir):
                os.mkdir(newdir)

        for entry in files:
            dest = normpath(joinpath(dstdir, root, entry))
            src = normpath(joinpath(srcdir, root, entry))
            if not isfile(dest) or not filecmp(src, dest):
                copyfile(src, dest)

    # some of the spec benchmarks expect to be run from one directory up.
    # just create some symlinks that solve the problem
    inlink = joinpath(dstdir, 'input')
    outlink = joinpath(dstdir, 'output')
    if not exists(inlink):
        os.symlink('.', inlink)
    if not exists(outlink):
        os.symlink('.', outlink)

class Benchmark(object):
    def __init__(self, isa, os, input_set):
        if not hasattr(self.__class__, 'name'):
            self.name = self.__class__.__name__

        if not hasattr(self.__class__, 'binary'):
            self.binary = self.name

        if not hasattr(self.__class__, 'args'):
            self.args = []

        if not hasattr(self.__class__, 'output'):
            self.output = '%s.out' % self.name

        if not hasattr(self.__class__, 'simpoint'):
            self.simpoint = None

        try:
            func = getattr(self.__class__, input_set)
        except AttributeError:
            raise AttributeError(
                'The benchmark %s does not have the %s input set' % \
                (self.name, input_set))

        executable = joinpath(spec_dist, 'binaries', self.binary)
        print ("executable= %s " % (executable))
        if not isfile(executable):
            raise AttributeError('%s not found' % executable)
        self.executable = executable

        # root of tree for input & output data files
        data_dir = joinpath(spec_dist, 'data', self.name)
        # optional subtree with files shared across input sets
        all_dir = joinpath(data_dir, 'all')
        # dirs for input & output files for this input set
        inputs_dir = joinpath(data_dir, input_set, 'input')
        outputs_dir = joinpath(data_dir, input_set, 'output')
        # keep around which input set was specified
        self.input_set = input_set
        print ("inputs_dir = %s " % (inputs_dir))
        print ("outputs_dir = %s " % (outputs_dir))
        if not isdir(inputs_dir):
            raise AttributeError('%s not found' % inputs_dir)

        self.inputs_dir = [ inputs_dir ]
        if isdir(all_dir):
            self.inputs_dir += [ joinpath(all_dir, 'input') ]
        if isdir(outputs_dir):
            self.outputs_dir = outputs_dir

        if not hasattr(self.__class__, 'stdin'):
            self.stdin = joinpath(inputs_dir, '%s.in' % self.name)
            if not isfile(self.stdin):
                self.stdin = None

        if not hasattr(self.__class__, 'stdout'):
            self.stdout = joinpath(outputs_dir, '%s.out' % self.name)
            if not isfile(self.stdout):
                self.stdout = None

        func(self, isa, os)

    def makeProcessArgs(self, **kwargs):
        # set up default args for Process object
        process_args = {}
        process_args['cmd'] = [ self.name ] + self.args
        process_args['executable'] = self.executable
        if self.stdin:
            process_args['input'] = self.stdin
        if self.stdout:
            process_args['output'] = self.stdout
        if self.simpoint:
            process_args['simpoint'] = self.simpoint
        # explicit keywords override defaults
        process_args.update(kwargs)

        return process_args

    def makeProcess(self, **kwargs):
        process_args = self.makeProcessArgs(**kwargs)

        # figure out working directory: use m5's outdir unless
        # overridden by Process's cwd param
        cwd = process_args.get('cwd')

        if not cwd:
            from m5 import options
            cwd = options.outdir
            process_args['cwd'] = cwd
        if not isdir(cwd):
            os.makedirs(cwd)
        # copy input files to working directory
        for d in self.inputs_dir:
            copyfiles(d, cwd)
        # generate Process object
        from m5.objects import Process
        return Process(**process_args)

    def __str__(self):
        return self.name

class DefaultBenchmark(Benchmark):
    def ref(self, isa, os): pass
    def test(self, isa, os): pass
    def train(self, isa, os): pass

class MinneDefaultBenchmark(DefaultBenchmark):
    def smred(self, isa, os): pass
    def mdred(self, isa, os): pass
    def lgred(self, isa, os): pass

class namd(DefaultBenchmark):
    name = 'namd'
    lang = 'C++'

    def all(self, isa, os):
        self.args = ['--input',f'{cpu2017_root}/data/namd/all/input/namd.input','--iterations', '1','--output','namd.out']
    def ref(self, isa, os):
        self.args = ['--input',f'{cpu2017_root}/data/namd/all/input/namd.input','--iterations', '1','--output','namd.out']

class blender(DefaultBenchmark):
    name = 'blender'
    lang = 'C++'

    def ref(self, isa, os):
        self.args = ['--input',f'{cpu2017_root}/data/blender/ref/input/sh3_no_char.blend']
        self.output = 'test.out'  

    def test(self, isa, os):
        self.args = ['--input',f'{cpu2017_root}/data/blender/test/input/cube.blend']
        self.output = 'ref.out'  

    def train(self, isa, os):
        self.args = ['-m1200',f'{cpu2017_root}/data/blender/train/input/sh5_reduced.blend']
        self.output = 'train.out' 
 
class bwaves(DefaultBenchmark):
    name = 'bwaves'
    number = 503
    lang = 'C'

    def all(self, isa, os):
        self.args = [ f'{cpu2017_root}/data/bwaves/test/' ]

class cactus(DefaultBenchmark):
    name = 'cactus'
    number = 507 
    lang = 'C++'

    def ref(self, isa, os):
        self.args = [f'{cpu2017_root}/data/cactus/ref/input/spec_ref.par']
        self.output = 'spec_ref.out'

class xalancbmk(DefaultBenchmark):
    name = 'xalancbmk'
    number =523
    lang = 'C++'

    def ref(self, isa, os):
    	self.args = [ '-v', f'{cpu2017_root}/data/xalancbmk/ref/input/t5.xml', f'{cpu2017_root}/data/xalancbmk/ref/input/xalanc.xsl']

class milc(DefaultBenchmark):
    name = 'milc'
    lang = 'C'
    def ref(self, isa, os):
        self.stdin = f'{cpu2017_root}/data/milc/ref/input/su3imp.in'


class soplex(DefaultBenchmark):
    name = 'soplex'
    lang = 'C++'

    def test(self, isa, os):
        self.args = ['-m10000',f'{cpu2017_root}/data/soplex/test/input/test.mps']
        self.output = 'test.out'  

    def ref(self, isa, os):
        self.args = ['-m3500',f'{cpu2017_root}/data/soplex/ref/input/ref.mps']
        self.output = 'ref.out'  

    def train(self, isa, os):
        self.args = ['-m1200',f'{cpu2017_root}/data/soplex/train/input/train.mps']
        self.output = 'train.out'  

class gamess(DefaultBenchmark):
    name = 'gamess'
    number = 416 
    lang = 'F95'

    def ref(self, isa, os):
     	self.stdin = f'{cpu2017_root}/data/gamess/ref/input/cytosine.2.config'

class bzip2(DefaultBenchmark):
    name = 'bzip2'
    number = 256
    lang = 'C'

    def all(self, isa, os):
        self.args = [f'{cpu2017_root}/data/bzip2/all/input/input.program', '1']

    def test(self, isa, os):
        self.args = [f'{cpu2017_root}/data/bzip2/test/input/dryer.jpg', '2']

    def train(self, isa, os):
        self.args = [f'{cpu2017_root}/data/bzip2/train/input/byoudoin.jpg', '5']

    def ref(self, isa, os):
        self.args = [f'{cpu2017_root}/data/bzip2/ref/input/input.source', '280']

class bzip2_source(bzip2):
    def ref(self, isa, os):
        self.simpoint = 977*100E6
        self.args = [ 'input.source', '58' ]

    def lgred(self, isa, os):
        self.args = [ 'input.source', '1' ]

class bzip2_graphic(bzip2):
    def ref(self, isa, os):
        self.simpoint = 718*100E6
        self.args = [ 'input.graphic', '58' ]

    def lgred(self, isa, os):
        self.args = [ 'input.graphic', '1' ]

class bzip2_program(bzip2):
    def ref(self, isa, os):
        self.simpoint = 458*100E6
        self.args = [ 'input.program', '58' ]

    def lgred(self, isa, os):
        self.args = [ 'input.program', '1' ]

class exchange(DefaultBenchmark):
    name = 'exchange'
    lang = 'C++'

    def ref(self, isa, os):
        self.args = ['--input', f'{cpu2017_root}/data/exchange/ref/input/control']
        self.output = 'test.out'

    def test(self, isa, os):
        self.args = ['--input', f'{cpu2017_root}/data/exchange/test/input/control']
        self.output = 'ref.out'

    def train(self, isa, os):
        self.args = ['-m1200', f'{cpu2017_root}/data/exchange/train/input/control']
        self.output = 'train.out'

class leela(DefaultBenchmark):
    name = 'leela'
    lang = 'C++'

    def ref(self, isa, os):
        self.args = ['--input', f'{cpu2017_root}/data/leela/ref/input/ref.sgf']
        self.output = 'test.out'

    def test(self, isa, os):
        self.args = ['--input', f'{cpu2017_root}/data/leela/test/input/test.sgf']
        self.output = 'ref.out'

    def train(self, isa, os):
        self.args = ['-m1200', f'{cpu2017_root}/data/leela/train/input/train.sgf']
        self.output = 'train.out'

class imagick(DefaultBenchmark):
    name = 'imagick'
    lang = 'C++'

    def ref(self, isa, os):
        self.args = ['--input', f'{cpu2017_root}/data/imagick/ref/input/control']
        self.output = 'test.out'

    def test(self, isa, os):
        self.args = ['--input', f'{cpu2017_root}/data/imagick/ref/input/control']
        self.output = 'ref.out'

    def train(self, isa, os):
        self.args = ['-m1200', f'{cpu2017_root}/data/imagick/ref/input/control']
        self.output = 'train.out'

class mcf(DefaultBenchmark):
    name = 'mcf'
    number = 181
    lang = 'C'

    def test(self, isa, os):
    	self.args = [ f'{cpu2017_root}/data/mcf/test/input/inp.in' ]

    def ref(self, isa, os):
    	self.args = [ f'{cpu2017_root}/data/mcf/ref/input/inp.in' ]

    def train(self, isa, os):
    	self.args = [ f'{cpu2017_root}/data/mcf/train/input/inp.in' ]
class omnetpp(DefaultBenchmark):
     name = 'omnetpp'
     lang = 'C++'
     def all(self, isa, os):
         self.args =  [f'{cpu2017_root}/data/omnetpp/ref/input/omnetpp.ini',
                     f'{cpu2017_root}/data/omnetpp/ref/output/omnetpp.log']
class parest(DefaultBenchmark):
     name = 'parest'
     lang = 'C++'
     def all(self, isa, os):
         self.args =  [f'{cpu2017_root}/data/parest/ref/input/ref.prm',
                     f'{cpu2017_root}/data/parest/ref/output/log']
class perlbench(DefaultBenchmark):
    name = 'perlbench'
    number = 400 
    lang = 'C'

    def all(self, isa, os):
        self.args = ['-I./lib', f'{cpu2017_root}/data/perlbench/ref/input/ref.prm',
                     f'{cpu2017_root}/data/perlbench/ref/output/log']

class povray(DefaultBenchmark):
    name = 'povray'
    number = 453 
    lang = 'C++'

    def ref(self, isa, os):
        self.args = [f'{cpu2017_root}/data/povray/ref/input/SPEC-benchmark-ref.ini',
                     f'{cpu2017_root}/data/povray/ref/output/povray.out']

class sjeng(DefaultBenchmark):
    name = 'sjeng'
    lang = 'C'
    
    def test(self, isa, os):
    	self.args = [f'{cpu2017_root}/data/sjeng/test/input/test.txt']

    def train(self, isa, os):
    	self.args = [f'{cpu2017_root}/data/sjeng/train/input/train.txt']

    def ref(self, isa, os):
    	self.args = [f'{cpu2017_root}/data/sjeng/ref/input/ref.txt']

class x264(DefaultBenchmark):
    name = 'x264'
    number = '464'
    lang = 'C'

    def ref(self, isa, os):
        self.args = [f'{cpu2017_root}/data/x264/ref/input/control']


class xz(DefaultBenchmark):
    name = 'xz'
    number = '464'
    lang = 'C'

    def ref(self, isa, os):
        self.args = [f'{cpu2017_root}/data/xz/ref/input/control']

    def all(self, isa, os):
        self.args = [f'{cpu2017_root}/data/xz/all/input/input.combined.xz']

    def test(self, isa, os):
        self.args = [f'{cpu2017_root}/data/xz/test/input/control']


class ammp(MinneDefaultBenchmark):
    name = 'ammp'
    number = 188
    lang = 'C'
    simpoint = 108*100E6

class applu(MinneDefaultBenchmark):
    name = 'applu'
    number = 173
    lang = 'F77'
    simpoint = 2179*100E6

class apsi(MinneDefaultBenchmark):
    name = 'apsi'
    number = 301
    lang = 'F77'
    simpoint = 3408*100E6

class art(DefaultBenchmark):
    name = 'art'
    number = 179
    lang = 'C'

    def test(self, isa, os):
        self.args = [ '-scanfile', 'c756hel.in',
                      '-trainfile1', 'a10.img',
                      '-stride', '2',
                      '-startx', '134',
                      '-starty', '220',
                      '-endx', '139',
                      '-endy', '225',
                      '-objects', '1' ]
        self.output = 'test.out'

    def train(self, isa, os):
        self.args = [ '-scanfile', 'c756hel.in',
                      '-trainfile1', 'a10.img',
                      '-stride', '2',
                      '-startx', '134',
                      '-starty', '220',
                      '-endx', '184',
                      '-endy', '240',
                      '-objects', '3' ]
        self.output = 'train.out'

    def lgred(self, isa, os):
        self.args = ['-scanfile', 'c756hel.in',
                     '-trainfile1', 'a10.img',
                     '-stride', '5',
                     '-startx', '134',
                     '-starty', '220',
                     '-endx', '184',
                     '-endy', '240',
                     '-objects', '1' ]
        self.output = 'lgred.out'


class art110(art):
    def ref(self, isa, os):
        self.args = [ '-scanfile', 'c756hel.in',
                      '-trainfile1', 'a10.img',
                      '-trainfile2', 'hc.img',
                      '-stride', '2',
                      '-startx', '110',
                      '-starty', '200',
                      '-endx', '160',
                      '-endy', '240',
                      '-objects', '10' ]
        self.output = 'ref.1.out'
        self.simpoint = 340*100E6

class art470(art):
    def ref(self, isa, os):
        self.args = [ '-scanfile', 'c756hel.in',
                      '-trainfile1', 'a10.img',
                      '-trainfile2', 'hc.img',
                      '-stride', '2',
                      '-startx', '470',
                      '-starty', '140',
                      '-endx', '520',
                      '-endy', '180',
                      '-objects', '10' ]
        self.output = 'ref.2.out'
        self.simpoint = 365*100E6

class equake(DefaultBenchmark):
    name = 'equake'
    number = 183
    lang = 'C'
    simpoint = 812*100E6

    def lgred(self, isa, os): pass

class facerec(MinneDefaultBenchmark):
    name = 'facerec'
    number = 187
    lang = 'F'
    simpoint = 375*100E6

class fma3d(MinneDefaultBenchmark):
    name = 'fma3d'
    number = 191
    lang = 'F'
    simpoint = 2541*100E6

class galgel(MinneDefaultBenchmark):
    name = 'galgel'
    number = 178
    lang = 'F'
    simpoint = 2491*100E6

class lucas(MinneDefaultBenchmark):
    name = 'lucas'
    number = 189
    lang = 'F'
    simpoint = 545*100E6

class mesa(Benchmark):
    name = 'mesa'
    number = 177
    lang = 'C'
    stdin = None

    def __set_args(self, frames):
        self.args = [ '-frames', frames, '-meshfile', '%s.in' % self.name,
                      '-ppmfile', '%s.ppm' % self.name ]

    def test(self, isa, os):
        self.__set_args('10')

    def train(self, isa, os):
        self.__set_args('500')

    def ref(self, isa, os):
        self.__set_args('1000')
        self.simpoint = 1135*100E6

    def lgred(self, isa, os):
        self.__set_args('1')

class mgrid(MinneDefaultBenchmark):
    name = 'mgrid'
    number = 172
    lang = 'F77'
    simpoint = 3292*100E6

class sixtrack(DefaultBenchmark):
    name = 'sixtrack'
    number = 200
    lang = 'F77'
    simpoint = 3043*100E6

    def lgred(self, isa, os): pass

class swim(MinneDefaultBenchmark):
    name = 'swim'
    number = 171
    lang = 'F77'
    simpoint = 2079*100E6

class wupwise(DefaultBenchmark):
    name = 'wupwise'
    number = 168
    lang = 'F77'
    simpoint = 3237*100E6

    def lgred(self, isa, os): pass

class bzip2(DefaultBenchmark):
    name = 'bzip2'
    number = 256
    lang = 'C'

    def test(self, isa, os):
        self.args = [ 'input.random' ]

    def train(self, isa, os):
        self.args = [ 'input.compressed' ]

class bzip2_source(bzip2):
    def ref(self, isa, os):
        self.simpoint = 977*100E6
        self.args = [ 'input.source', '58' ]

    def lgred(self, isa, os):
        self.args = [ 'input.source', '1' ]

class bzip2_graphic(bzip2):
    def ref(self, isa, os):
        self.simpoint = 718*100E6
        self.args = [ 'input.graphic', '58' ]

    def lgred(self, isa, os):
        self.args = [ 'input.graphic', '1' ]

class bzip2_program(bzip2):
    def ref(self, isa, os):
        self.simpoint = 458*100E6
        self.args = [ 'input.program', '58' ]

    def lgred(self, isa, os):
        self.args = [ 'input.program', '1' ]
class lbm(DefaultBenchmark):
    name  = 'lbm'
    lang = 'C'
    
    def test(self, isa, os):
         self.args = ['20', 'reference.dat', '0', '1' , f'{cpu2017_root}/data/lbm/test/input/100_100_130_cf_a.of']

    def train(self, isa, os):
         self.args = ['300', 'reference.dat', '0', '1' , f'{cpu2017_root}/data/lbm/train/input/100_100_130_cf_b.of']

    def ref(self, isa, os):
         self.args = ['3000', 'reference.dat', '0', '0' , f'{cpu2017_root}/data/lbm/ref/input/100_100_130_ldc.of']

class crafty(MinneDefaultBenchmark):
    name = 'crafty'
    number = 186
    lang = 'C'
    simpoint = 774*100E6

class eon(MinneDefaultBenchmark):
    name = 'eon'
    number = 252
    lang = 'CXX'
    stdin = None

class eon_kajiya(eon):
    args = [ 'chair.control.kajiya', 'chair.camera', 'chair.surfaces',
             'chair.kajiya.ppm', 'ppm', 'pixels_out.kajiya']
    output = 'kajiya_log.out'


class eon_cook(eon):
    args = [ 'chair.control.cook', 'chair.camera', 'chair.surfaces',
             'chair.cook.ppm', 'ppm', 'pixels_out.cook' ]
    output = 'cook_log.out'

class eon_rushmeier(eon):
    args = [ 'chair.control.rushmeier', 'chair.camera', 'chair.surfaces',
             'chair.rushmeier.ppm', 'ppm', 'pixels_out.rushmeier' ]
    output = 'rushmeier_log.out'
    simpoint = 403*100E6

class gap(DefaultBenchmark):
    name = 'gap'
    number = 254
    lang = 'C'

    def __set_args(self, size):
        self.args = [ '-l', './', '-q', '-m', size ]

    def test(self, isa, os):
        self.__set_args('64M')

    def train(self, isa, os):
        self.__set_args('128M')

    def ref(self, isa, os):
        self.__set_args('192M')
        self.simpoint = 674*100E6

    def lgred(self, isa, os):
        self.__set_args('64M')

    def mdred(self, isa, os):
        self.__set_args('64M')

    def smred(self, isa, os):
        self.__set_args('64M')

class gcc(DefaultBenchmark):
    name = 'gcc'
    number = 176
    lang = 'C'

    def test(self, isa, os):
        self.args = [ 'cccp.i', '-o', 'cccp.s' ]

    def train(self, isa, os):
        self.args = [ 'cp-decl.i', '-o', 'cp-decl.s' ]

    def smred(self, isa, os):
        self.args = [ 'c-iterate.i', '-o', 'c-iterate.s' ]

    def mdred(self, isa, os):
        self.args = [ 'rdlanal.i', '-o', 'rdlanal.s' ]

    def lgred(self, isa, os):
        self.args = [ 'cp-decl.i', '-o', 'cp-decl.s' ]

class gcc_166(gcc):
    def ref(self, isa, os):
        self.simpoint = 389*100E6
        self.args = [ '166.i', '-o', '166.s' ]

class gcc_200(gcc):
    def ref(self, isa, os):
        self.simpoint = 736*100E6
        self.args = [ '200.i', '-o', '200.s' ]

class gcc_expr(gcc):
    def ref(self, isa, os):
        self.simpoint = 36*100E6
        self.args = [ 'expr.i', '-o', 'expr.s' ]

class gcc_integrate(gcc):
    def ref(self, isa, os):
        self.simpoint = 4*100E6
        self.args = [ 'integrate.i', '-o', 'integrate.s' ]

class gcc_scilab(gcc):
    def ref(self, isa, os):
        self.simpoint = 207*100E6
        self.args = [ 'scilab.i', '-o', 'scilab.s' ]

class gzip(DefaultBenchmark):
    name = 'gzip'
    number = 164
    lang = 'C'

    def test(self, isa, os):
        self.args = [ 'input.compressed', '2' ]

    def train(self, isa, os):
        self.args = [ 'input.combined', '32' ]

class gzip_source(gzip):
    def ref(self, isa, os):
        self.simpoint = 334*100E6
        self.args = [ 'input.source', '1' ]
    def smred(self, isa, os):
        self.args = [ 'input.source', '1' ]
    def mdred(self, isa, os):
        self.args = [ 'input.source', '1' ]
    def lgred(self, isa, os):
        self.args = [ 'input.source', '1' ]

class gzip_log(gzip):
    def ref(self, isa, os):
        self.simpoint = 265*100E6
        self.args = [ 'input.log', '60' ]
    def smred(self, isa, os):
        self.args = [ 'input.log', '1' ]
    def mdred(self, isa, os):
        self.args = [ 'input.log', '1' ]
    def lgred(self, isa, os):
        self.args = [ 'input.log', '1' ]

class gzip_graphic(gzip):
    def ref(self, isa, os):
        self.simpoint = 653*100E6
        self.args = [ 'input.graphic', '60' ]
    def smred(self, isa, os):
        self.args = [ 'input.graphic', '1' ]
    def mdred(self, isa, os):
        self.args = [ 'input.graphic', '1' ]
    def lgred(self, isa, os):
        self.args = [ 'input.graphic', '1' ]

class gzip_random(gzip):
    def ref(self, isa, os):
        self.simpoint = 623*100E6
        self.args = [ 'input.random', '60' ]
    def smred(self, isa, os):
        self.args = [ 'input.random', '1' ]
    def mdred(self, isa, os):
        self.args = [ 'input.random', '1' ]
    def lgred(self, isa, os):
        self.args = [ 'input.random', '1' ]

class gzip_program(gzip):
    def ref(self, isa, os):
        self.simpoint = 1189*100E6
        self.args = [ 'input.program', '60' ]
    def smred(self, isa, os):
        self.args = [ 'input.program', '1' ]
    def mdred(self, isa, os):
        self.args = [ 'input.program', '1' ]
    def lgred(self, isa, os):
        self.args = [ 'input.program', '1' ]

class mcf(MinneDefaultBenchmark):
    name = 'mcf'
    number = 181
    lang = 'C'
    args = [ 'mcf.in' ]
    simpoint = 553*100E6

class parser(MinneDefaultBenchmark):
    name = 'parser'
    number = 197
    lang = 'C'
    args = [ '2.1.dict', '-batch' ]
    simpoint = 1146*100E6

class perlbmk(DefaultBenchmark):
    name = 'perlbmk'
    number = 253
    lang = 'C'

    def test(self, isa, os):
        self.args = [ '-I.', '-I', 'lib', 'test.pl' ]
        self.stdin = 'test.in'

class perlbmk_diffmail(perlbmk):
    def ref(self, isa, os):
        self.simpoint = 141*100E6
        self.args = [ '-I', 'lib', 'diffmail.pl', '2', '550', '15', '24',
                      '23', '100' ]

    def train(self, isa, os):
        self.args = [ '-I', 'lib', 'diffmail.pl', '2', '350', '15', '24',
                      '23', '150' ]

class perlbmk_scrabbl(perlbmk):
    def train(self, isa, os):
        self.args = [ '-I.', '-I', 'lib', 'scrabbl.pl' ]
        self.stdin = 'scrabbl.in'

class perlbmk_makerand(perlbmk):
    def ref(self, isa, os):
        self.simpoint = 11*100E6
        self.args = [ '-I', 'lib',  'makerand.pl' ]

    def lgred(self, isa, os):
        self.args = [ '-I.', '-I', 'lib', 'lgred.makerand.pl' ]

    def mdred(self, isa, os):
        self.args = [ '-I.', '-I', 'lib', 'mdred.makerand.pl' ]

    def smred(self, isa, os):
        self.args = [ '-I.', '-I', 'lib', 'smred.makerand.pl' ]

class perlbmk_perfect(perlbmk):
    def ref(self, isa, os):
        self.simpoint = 5*100E6
        self.args = [ '-I', 'lib',  'perfect.pl', 'b', '3', 'm', '4' ]

    def train(self, isa, os):
        self.args = [ '-I', 'lib', 'perfect.pl', 'b',  '3' ]

class perlbmk_splitmail1(perlbmk):
    def ref(self, isa, os):
        self.simpoint = 405*100E6
        self.args = [ '-I', 'lib', 'splitmail.pl', '850', '5', '19',
                      '18', '1500' ]

class perlbmk_splitmail2(perlbmk):
    def ref(self, isa, os):
        self.args = [ '-I', 'lib', 'splitmail.pl', '704', '12', '26',
                      '16', '836' ]

class perlbmk_splitmail3(perlbmk):
    def ref(self, isa, os):
        self.args = [ '-I', 'lib', 'splitmail.pl', '535', '13', '25',
                      '24', '1091' ]

class perlbmk_splitmail4(perlbmk):
    def ref(self, isa, os):
        self.args = [ '-I', 'lib', 'splitmail.pl', '957', '12', '23',
                      '26', '1014' ]

class twolf(Benchmark):
    name = 'twolf'
    number = 300
    lang = 'C'
    stdin = None

    def test(self, isa, os):
        self.args = [ 'test' ]

    def train(self, isa, os):
        self.args = [ 'train' ]

    def ref(self, isa, os):
        self.simpoint = 1066*100E6
        self.args = [ 'ref' ]

    def smred(self, isa, os):
        self.args = [ 'smred' ]

    def mdred(self, isa, os):
        self.args = [ 'mdred' ]

    def lgred(self, isa, os):
        self.args = [ 'lgred' ]

class vortex(Benchmark):
    name = 'vortex'
    number = 255
    lang = 'C'
    stdin = None

    def __init__(self, isa, os, input_set):
        if (isa in ('arm', 'thumb', 'aarch64')):
            self.endian = 'lendian'
        elif (isa == 'sparc' or isa == 'sparc32'):
            self.endian = 'bendian'
        else:
            raise AttributeError("unknown ISA %s" % isa)

        super(vortex, self).__init__(isa, os, input_set)

    def test(self, isa, os):
        self.args = [ '%s.raw' % self.endian ]
        self.output = 'vortex.out'

    def train(self, isa, os):
        self.args = [ '%s.raw' % self.endian ]
        self.output = 'vortex.out'

    def smred(self, isa, os):
        self.args = [ '%s.raw' % self.endian ]
        self.output = 'vortex.out'

    def mdred(self, isa, os):
        self.args = [ '%s.raw' % self.endian ]
        self.output = 'vortex.out'

    def lgred(self, isa, os):
        self.args = [ '%s.raw' % self.endian ]
        self.output = 'vortex.out'

class vortex1(vortex):
    def ref(self, isa, os):
        self.args = [ '%s1.raw' % self.endian ]
        self.output = 'vortex1.out'
        self.simpoint = 271*100E6


class vortex2(vortex):
    def ref(self, isa, os):
        self.simpoint = 1024*100E6
        self.args = [ '%s2.raw' % self.endian ]
        self.output = 'vortex2.out'

class vortex3(vortex):
    def ref(self, isa, os):
        self.simpoint = 564*100E6
        self.args = [ '%s3.raw' % self.endian ]
        self.output = 'vortex3.out'

class vpr(MinneDefaultBenchmark):
    name = 'vpr'
    number = 175
    lang = 'C'

# not sure about vpr minnespec place.in
class vpr_place(vpr):
    args = [ 'net.in', 'arch.in', 'place.out', 'dum.out', '-nodisp',
             '-place_only', '-init_t', '5', '-exit_t', '0.005',
             '-alpha_t', '0.9412', '-inner_num', '2' ]
    output = 'place_log.out'

class vpr_route(vpr):
    simpoint = 476*100E6
    args = [ 'net.in', 'arch.in', 'place.in', 'route.out', '-nodisp',
             '-route_only', '-route_chan_width', '15',
             '-pres_fac_mult', '2', '-acc_fac', '1',
             '-first_iter_pres_fac', '4', '-initial_pres_fac', '8' ]
    output = 'route_log.out'

all = [namd,blender,bwaves,cactus,xalancbmk,exchange,gcc,lbm,imagick,leela,mcf,omnetpp,parest,perlbench,povray,sjeng,x264,xz]

__all__ = [ x.__name__ for x in all ]

if __name__ == '__main__':
    from pprint import pprint
    for bench in all:
        for input_set in 'ref', 'test', 'train':
            print('class: %s' % bench.__name__)
            x = bench('x86', 'linux', input_set)
            print('%s: %s' % (x, input_set))
            pprint(x.makeProcessArgs())
            print()
