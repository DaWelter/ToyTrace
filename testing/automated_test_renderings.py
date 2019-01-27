#!/usr/bin/env python
# coding: utf-8

import os, sys
import subprocess
from math import pi
import numpy as np
from collections import OrderedDict, defaultdict
import tempfile
import datetime
import hashlib
import textwrap
from PIL import Image, ImageDraw, ImageFont
import logging
import functools
import fnmatch
import itertools
import copy
import argparse
# References for PIL:
# http://pillow.readthedocs.io/en/5.0.0/reference/ImageDraw.html
# https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python

destination_dir =     os.path.join('renders-'+datetime.datetime.now().strftime('%d-%m-%Y-%H'))
if not os.path.isdir(destination_dir):
    os.mkdir(destination_dir)
scenefile_dir = os.path.join(destination_dir, 'generated_scenes')
the_exe = '' #'../buildopt/toytrace'

def merge(d1, d2):
    return dict(d1.items() + d2.items())


def logRendering(name, opts):
    if "--algo" in opts:
        mode = opts['--algo']
    else:
        mode = 'pt'
    logging.info("---------  Render: '%s' (%s) -----------", name, mode)


def toytrace(scene, output_file, opts, name = "UNKNOWN"):
    logRendering(name, opts)
    use_tmpfile = not output_file
    if use_tmpfile:
        output_file = tempfile.mktemp(suffix='.png', prefix='/tmp/')
    else:
        output_file = os.path.join(destination_dir, output_file)
    opts = sum([[k, str(v)] for k,v in opts.items()], [])
    if os.path.isfile(os.path.join('scenes',scene)):
        input_file = os.path.join('scenes',scene)
    else:
        input_file = '-'
    cmd = [the_exe, '--no-display', '-I', 'scenes', '--input-file', input_file, '--output-file', output_file]
    cmd += opts
    print cmd
    #current_dir = os.getcwd()
    #os.chdir('scenes')

    if os.path.isfile(scene):
        p = subprocess.Popen(cmd)
        p.communicate()
    else:
        p = subprocess.Popen(cmd, stdin = subprocess.PIPE)
        stdout, stderr = p.communicate(input = scene)
    #os.chdir(current_dir)
    if p.wait() != 0:
        raise RuntimeError("Toytrace failed!")
    if use_tmpfile:
        return Image.open(output_file)


class SceneCollection(object):
    def __init__(self, name_prefix):
        self._name_prefix = name_prefix
        self._scenes = {}

    def __setitem__(self, key, value):
        self._scenes[key] = value

    def __getitem__(self, key):
        return self._scenes[key]

    def items(self):
        return self._scenes.items()

    def dumpToSceneDir(scenes):
        filename_prefix = scenes._name_prefix
        if not os.path.isdir(scenefile_dir):
            os.mkdir(scenefile_dir)
        for k, scene in scenes.items():
            if hasattr(scene, '__call__') and not isinstance(scene, str):
                scene = scene()
            filename = os.path.join(scenefile_dir, filename_prefix + k + '.nff')
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    target_checksum = hashlib.md5(f.read()).digest()
                checksum = hashlib.md5(scene).digest()
                if target_checksum != checksum:
                    raise RuntimeError(
                        "Attempt to overwrite %s which is different from what would be written!" % filename)
            else:
                with open(filename, 'w') as f:
                    f.write(scene)
            scenes[k] = filename


def dumpToSceneDir(scenes):
    if not os.path.isdir(scenefile_dir):
        os.mkdir(scenefile_dir)
    for k, scene in scenes:
        if hasattr(scene, '__call__') and not isinstance(scene, str):
            scene = scene()
        filename = os.path.join(scenefile_dir, k + '.nff')
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                target_checksum = hashlib.md5(f.read()).digest()
            checksum = hashlib.md5(scene).digest()
            if target_checksum != checksum:
                raise RuntimeError(
                    "Attempt to overwrite %s which is different from what would be written!" % filename)
        else:
            with open(filename, 'w') as f:
                f.write(scene)


class CornelBoxLightTypes(object):
    def __init__(self):
        self.buildScenePieces()
        self.area_light_test_template = """
        v
        from 0 0.5 1.4
        at 0 0.5 0
        up 0 1 0
        resolution 640 480
        angle 40

        {light_spec}

        diffuse white  1 1 1 0.8
        diffuse red    1 0 0 0.8
        diffuse green  0 1 0 0.8
        diffuse blue   1 1 1 0.8

        # reset transform
        transform

        m cornelbox.dae

        glossy spheremat 1 1 1 0.8 0.1
        s 0.3 0.2 -0.5 0.2

        diffuse boxmat 1 1 1 0.4
        m box_in_cornelbox.dae
        """
        self.W = 320
        self.H = 320
        self.common_opt = { '-w' : self.W, '-h' : self.H}

    def buildScenePieces(self):
        self.light_specs = OrderedDict()
        # Reference for radiometric quantities: https://en.wikipedia.org/wiki/Radiometry
        radiant_flux = 4.  # In Watt.
        self.light_specs['point_light'] = """
        l 0 0.8 -0.5  1. 1. 1. %f
        """ % radiant_flux
        #print "Point light flux =", radiant_flux

        # Uniform area lights are specified in terms of Irradiance, i.e. (Spectral) power per area.
        radius = 0.001
        area = (4.*pi*radius**2)
        irradiance = radiant_flux / area
        self.light_specs['small_sphere_light'] = """
        larea arealight1 uniform 1. 1. 1. %f
        diffuse black  1 1 1 0.
        s 0 0.8 -0.5 %f
        larea none
        """ % (irradiance, radius)
        #print "Small sphere irradiance", irradiance

        radius = 0.1
        area = (4.*pi*radius**2)
        irradiance = radiant_flux / area
        self.light_specs['big_sphere_light'] = """
        larea arealight1 uniform 1. 1. 1. %f
        diffuse black  1 1 1 0.
        s 0 0.8 -0.5 %f
        larea none
        """ % (irradiance, radius)
        #print "Big sphere irradiance", irradiance

        size = 0.05
        area = (6.*size*size)
        irradiance = radiant_flux / area
        self.light_specs['small_cube_light'] = """
        larea arealight1 uniform 1. 1. 1. %f
        diffuse black  1 1 1 0.
        transform 0 0.8 -0.5 0 0 0 %f %f %f
        m unitcube.dae
        larea none
        """ % (irradiance, size, size, size)
        #print "Small cube irradiance", irradiance

        size = 0.25
        area = (6.*size*size)
        irradiance = radiant_flux / area
        self.light_specs['big_cube_light'] = """
        larea arealight1 uniform 1. 1. 1. %f
        diffuse black  1 1 1 0.
        transform 0 0.8 -0.5 0 0 0 %f %f %f
        m unitcube.dae
        larea none
        """ % (irradiance, size, size, size)
        #print "Big cube irradiance", irradiance


    def makeScene(self, l):
        return textwrap.dedent(self.area_light_test_template.format(**{'light_spec': self.light_specs[l]}))

    def makeScenes(self):
        for k in self.light_specs.keys():
            name = "cornelbox_lighttest_"+k
            yield name, functools.partial(self.makeScene, k)

    def test_forwardtraced(self):
        W, H = self.W, self.H
        imagegrid = []
        for name, scene in self.makeScenes():
            scene = scene()
            image_row = []
            for mode in 'both bsdf lights'.split():
                img = toytrace(
                    scene, '',
                    merge(self.common_opt, { '--pt-sample-mode' : mode, '--spp' : 16, }),
                    name = name)
                image_row.append((name, mode, img))
            imagegrid.append(image_row)
        new_im = Image.new('RGB', (W*3, 10+H*len(imagegrid)))
        d = ImageDraw.Draw(new_im)
        for i, row in enumerate(imagegrid):
            for j, (name, mode, img) in enumerate(row):
                new_im.paste(img, (j*W, 10+i*H))
                if i == 0:
                    d.text((1 + j*W, 1), mode)
                if j == 0:
                    d.text((1, 11+i*H), name)
        new_im.save(os.path.join(destination_dir, 'arealights.png'))


    def test_bdpt(self):
        W, H = self.W, self.H
        images = []
        for name, scene in self.makeScenes():
            scene = scene()
            img = toytrace(
                scene, '',
                merge(self.common_opt, {'--algo': 'bdpt', '--spp' : 4 }),
                name = name)
            images.append((name, img))
        new_im  = Image.new('RGB', (W, H*len(images)))
        d = ImageDraw.Draw(new_im)
        for i, (name, im) in enumerate(images):
            new_im.paste(im, (0, i*H))
            d.text((1, i*H), name)
        new_im.save(os.path.join(destination_dir, 'arealights_bdpt.png'))

    def makeTests(self):
      yield ('test_forwardtraced', self.test_forwardtraced)
      yield ('test_bdpt', self.test_bdpt)


class EmissiveDemoMediumScenes(object):
    def __init__(self):
      self.template = """
        v
        from 0 0.5 1.4
        at 0 0.5 0
        up 0 1 0
        resolution 480 480
        angle 40


        {{
        transform 0 1.00 -0.55 0 0 0 0.3 0.02 0.3
        diffuse verywhite 1 1 1 0.9
        larea arealight2 uniform 1 1 1 {}
        m unitcube.dae
        }}

        diffuse white  1 1 1 0.5
        diffuse red    1 0 0 0.5
        diffuse green  0 1 0 0.5
        diffuse blue   1 1 1 0.5

        m cornelbox.dae

        glossy spheremat 1 1 1 0.8 0.1
        s 0.3 0.2 -0.8 0.2

        shader white
        transform 0 0 0 0 0 0 1 1 1
        m box_in_cornelbox.dae

        shader invisible
        emissivedemomedium m1 {} {} 1 {} 0 0.5 -0.5 0.2
        transform 0 0.5 -0.5 0 0 0 0.4 0.4 0.4
        m unitcube.dae
      """
      self.cross_sections = [0.1, 1., 10.]
      self.temperatures = [ 100., 3000., 6000.]
      self.main_light_powers = [ 100., 100., 1. ]

    def _makeScene(self, params):
        cross_section, temperature, main_light_power = params
        return textwrap.dedent(self.template.format(main_light_power, cross_section, cross_section, temperature))

    def makeScenes(self):
        for cross_section in self.cross_sections:
            for main_light_power, temperature in zip(self.main_light_powers, self.temperatures):
                name = "emissivedemomedium_{}K_e{}".format(int(temperature), int(cross_section*10))
                yield name, functools.partial(self._makeScene, (cross_section, temperature, main_light_power))

    def _run(self, scene, name, render_mode):
        #opt = {'-w': 512, '-h': 512, '--spp': 512,
        opt = {'-w': 256, '-h': 256, '--spp': 32,
               '--algo' : render_mode }
        toytrace(
            scene, name+'.png', opt, name=name)

    def makeTests(self):
        for prefix, scene_builder in self.makeScenes():
            scene = scene_builder()
            for render_mode in ['pt', 'bdpt']:
                name = '{}_{}'.format(prefix, render_mode)
                yield (name, functools.partial(self._run, scene, name, render_mode))



class Atmosphere(object):
    def __init__(self):
        self.scene_template = """
        {camera_spec}

        {light_spec}

        include atmosphere.nff
        """
        self.light_specs = {
            'zenit': 'lsun 0 0 -1 100',
            'late': 'lsun -1 0 -0.02 100',
            'evening': 'lsun -1 0 -1 100',
            'later': 'lsun -1 0 0.087 5000  # -5 deg'
        }
        self.cam_specs = {
            'perspective': """
        v
        from 0 0 6300.1
        at 1 1 6301
        up 0 0 1
        resolution 640 480
        angle 120
        """,
            'fisheye': """
        vfisheye
        from 0 0 6300.01
        at 0 0 6301
        up 1 0 0
        resolution 640 640
        """
        }

        self.spacescene = textwrap.dedent("""
        v
        from -300000 -200000 0
        at 0 0 0
        up 0 1 0
        resolution 640 640
        angle 2.2

        lsun 3 0.7 -0.5 200

        include atmosphere_textured.nff
        """)
        self.backlit = textwrap.dedent("""
        v
        from 0 0 6340
        at -10 0 6339
        up 0 0 1
        resolution 640 640
        angle 20

        lsun 1 0 0.1 200

        include atmosphere_textured.nff
        """)

    def makeGroundScene(self, light_key, cam_key):
        return textwrap.dedent(self.scene_template.format(**{'light_spec': self.light_specs[light_key], 'camera_spec': self.cam_specs[cam_key]}))

    def makeScenes(self):
        for k, v in self.light_specs.items():
            for l, w in self.cam_specs.items():
                name = 'atmosphere_%s_%s' % (k ,l)
                yield (name, functools.partial(self.makeGroundScene, k, l))
        yield ('atmosphere_space', lambda : self.spacescene)
        yield ('atmosphere_backlit', lambda : self.backlit)

    def makeTests(self):
        common_opt = {'--spp': 16, '-w': 320, '-h': 320}
        for name, scene in self.makeScenes():
            func = functools.partial(toytrace, scene(), name+'.png', common_opt, name = name)
            yield (name, func)




class Various(object):
    def __init__(self):
        self.scenes = {}
        self.scenes['various_veach_mis_scene'] = 'veach_mis_scene.nff'
        self.scenes['various_veach_mis_scene_distant'] = 'veach_mis_scene_distant.nff'
        self.scenes['various_material_test_scene'] = 'material_test_scene.nff'
        self.scenes['various_cornelbox_parabolic_reflector'] = 'cornelbox_parabolic_reflector.nff'
        self.scenes['various_env_sphere_mirror'] = 'env_sphere_mirror.nff'
        self.scenes['various_env_sphere_mirror_plus_area'] = 'env_sphere_mirror_plus_area.nff'
        self.scenes['various_litcorner'] = 'litcorner.nff'
        self.scenes['various_litcorner2'] = 'litcorner2.nff'
        self.scenes['various_cubelight'] = 'cubelight.nff'
        self.scenes['various_texture_test'] = 'texture_test.nff'
        self.scenes['various_off_center_rgb_cmy'] = 'off_center_rgb_cmy.nff'
        self.scenes['various_veach_shading_normals'] = 'veach_shading_normals.nff'
        self.scenes['various_test_lowpoly_sphere'] = 'test_lowpoly_sphere.nff'
        self.scenes['various_refractive_sphere_caustics'] = 'refractive_sphere_caustics.nff'
        self.scenes['various_refract_beam_at_cube'] = 'refract_beam_at_cube.nff'
        self.scenes['various_refractive_sphere_caustics_wo_medium'] = 'refractive_sphere_caustics_wo_medium.nff'
        # self.common_opt = defaultdict(lambda : {'-w': 480, '-h': 320})
        # self.common_opt['various_cornel_parallel_arealight'] = {'-w': 320, '-h': 320}
        # self.common_opt['various_cornelbox_parabolic_reflector'] = {'-w': 320, '-h': 320}

    def runSingle(self, name, scene, mode):       
        if mode == 'bdpt':
          opts = { '--algo' : 'bdpt', '--spp': 8}
          filename = name+'_bdpt.png'
        else:
          opts = {'--spp': 16}
          filename = name+'.png'
        toytrace(
            self.scenes[name], filename,
            opts,
            name = name)

    def makeTests(self):
        for name, scene in self.scenes.items():
          yield (name, functools.partial(self.runSingle, name, scene, 'pt'))
          yield (name+'_bdpt', functools.partial(self.runSingle, name, scene, 'bdpt'))


class ParticipatingMediaSimple(object):
    def __init__(self):
        self.pattern = '''
        v
        from 0 0 -10
        at 0 0 0
        up 0 1 0
        resolution 256 256
        angle 7

        {{
        diffuse diff 1 1 1 0.9
        transform 0 -0.55 0 0 1.57 0 2 2 2
        p 4
        1 1 0
        -1 1 0
        -1 -1 0
        1 -1 0
        }}

        {occlusion_spec}

        {light_spec}

        shader invisible
        {medium_spec}
        m unitcube.dae
        '''
        self.light_specs = dict([
        ('dir', "lddirection 0 -1 0 2 2 2"),
        ('area', """
        {
        transform 0 0.501 0 0 0 0 2 2 2
        larea thelight uniform 1 1 1 10
        p 4
        0.5 0 0.5
        -0.5 0 0.5
        -0.5 0 -0.5
        0.5 0 -0.5
        }"""),
        ('point', 'l 0 0.8 0  1. 1. 1. 30'),
        ('parallel', """
        {
        transform 0 0.55 0 0 0 0 2 2 2
        larea thelight parallel 1 1 1 10
        p 4
        0.5 0 0.5
        -0.5 0 0.5
        -0.5 0 -0.5
        0.5 0 -0.5
        }"""),
        ])
        self.occlusion = '''
        {
        diffuse diff 1 1 1 0
        transform -0.6 0.52 0 0 0 0 1 0.01 2
        m unitcube.dae
        transform 0.6 0.52 0 0 0 0 1 0.01 2
        m unitcube.dae
        }
        '''
        self.occluded_light_spec = self.light_specs.copy()
        self.occluded_light_spec['area'] = '''
        {
        transform 0 0.54 0 0 0 0 2 2 2
        larea arealight2 uniform 1 1 1 10
        p 4
        0.5 0 0.5
        -0.5 0 0.5
        -0.5 0 -0.5
        0.5 0 -0.5
        }
        '''
        self.media_specs = dict([
            ('dense', 'medium prettydense 2 2'),
            ('light', 'medium light 0.2 0.2'),
            ('scatter', 'medium scatter 10 0.1'),
            ('red', 'medium dense 4 0 0 0 0 4 4'),
            ('colorful', 'medium dense 2 0 0 0 2 0 0'),
        ])

    def assemble_prefix(self, config):
        light_spec, media_spec, with_occlusion = config
        return 'simplemedia_{}_{}_{}'.format(light_spec, media_spec, 'oc' if with_occlusion else 'wo')

    def assemble_scene(self, config):
        light_spec, media_spec, with_occlusion = config
        light_part = self.occluded_light_spec[light_spec] if with_occlusion else self.light_specs[light_spec]
        media_part = self.media_specs[media_spec]
        occlusion_part = self.occlusion if with_occlusion else ''
        config = {'light_spec': light_part, 'medium_spec': media_part, 'occlusion_spec' : occlusion_part}
        scene = textwrap.dedent(self.pattern.format(**config))
        return scene

    def run(self, scene, name, render_mode):
        opt = {'-w': 256, '-h': 256, '--spp': 16,
               '--algo' : render_mode }
        toytrace(
            scene, name+'.png', opt, name=name)

    def makeScenes(self):
        for light_spec in self.light_specs.keys():
            for media_spec in self.media_specs.keys():
                for with_occlusion in [True, False]:
                    config = (light_spec, media_spec, with_occlusion)
                    name = self.assemble_prefix(config)
                    yield (name, functools.partial(self.assemble_scene, config))

    def makeTests(self):
        for prefix, scene_builder in self.makeScenes():
            scene = scene_builder()
            for render_mode in ['pt', 'bdpt']:
                name = '{}_{}'.format(prefix, render_mode)
                yield (name, functools.partial(self.run, scene, name, render_mode))



if __name__ == '__main__':
    logging.getLogger().setLevel("INFO")
    parser = argparse.ArgumentParser(description='Render the test scenes.')
    parser.add_argument('--dump', action='store_true', default = False)
    parser.add_argument('--list', action='store_true', default = False)
    parser.add_argument('--exe', type = str, default = the_exe)
    parser.add_argument('pattern', default = '*', type = str, nargs = '?')
    args = parser.parse_args()

    def FilterFunc(stuff):
        name, _ = stuff
        result = fnmatch.fnmatch(name, args.pattern)
        return result

    the_classes = [
        ParticipatingMediaSimple,
        Various,
        Atmosphere,
        CornelBoxLightTypes,
        EmissiveDemoMediumScenes
    ]

    the_exe = args.exe
    if not os.path.isfile(the_exe):
        print "Cannot find toytrace exe: ", the_exe
        sys.exit(-1)

    if args.dump:
        the_generators = [ c().makeScenes() for c in the_classes if hasattr(c, "makeScenes") ]
        the_things = itertools.chain(*the_generators)
        to_execute = filter(FilterFunc, the_things)
        if args.list:
            for name, scene_gen in to_execute:
                print name
        else:
            dumpToSceneDir(to_execute)
    else:
        the_generators = [c().makeTests() for c in the_classes]
        the_things = itertools.chain(*the_generators)
        to_execute = filter(FilterFunc, the_things)
        for name, t in to_execute:
            if args.list:
                print name
            else:
                t()
