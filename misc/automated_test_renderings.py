#!/bin/env/python
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
import unittest
import logging
# References for PIL:
# http://pillow.readthedocs.io/en/5.0.0/reference/ImageDraw.html
# https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python


os.chdir('../scenes')
destination_dir =     os.path.join('..', 'misc', 'renders-'+datetime.datetime.now().strftime('%d-%m-%Y-%H'))
if not os.path.isdir(destination_dir):
    os.mkdir(destination_dir)
scenefile_dir = os.path.join(destination_dir, 'scenes')


def merge(d1, d2):
    return dict(d1.items() + d2.items())


def logRendering(name, opts):
    if "--algo" in opts:
        mode = opts['--algo']
    else:
        mode = 'pt'
    logging.info("Render: '%s' (%s)", name, mode)


def toytrace(scene, output_file, opts, name = "UNKNOWN"):
    logRendering(name, opts)
    use_tmpfile = not output_file
    if use_tmpfile:
        output_file = tempfile.mktemp(suffix='.png', prefix='/tmp/')
    else:
        output_file = os.path.join(destination_dir, output_file)
    opts = sum([[k, str(v)] for k,v in opts.items()], [])
    if os.path.isfile(scene):
        input_file = scene
    else:
        input_file = '-'
    cmd = ['../buildopt/toytrace', '--no-display', '--input-file', input_file, '--output-file', output_file]
    cmd += opts
    #print cmd
    if os.path.isfile(scene):
        p = subprocess.Popen(cmd)
        p.communicate()
    else:
        p = subprocess.Popen(cmd, stdin = subprocess.PIPE)
        stdout, stderr = p.communicate(input = scene)        
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



class CornelBoxLightTypesScenes(SceneCollection):
    def __init__(self):
        SceneCollection.__init__(self, 'cornelbox_')
        self.buildScenePieces()
        self.buildScenes()

    def buildScenePieces(self):
        self.light_specs = OrderedDict()
        # Reference for radiometric quantities: https://en.wikipedia.org/wiki/Radiometry
        radiant_flux = 4.  # In Watt.
        self.light_specs['point_light'] = """
        l 0 0.8 -0.5  1. 1. 1. %f
        """ % radiant_flux
        print "Point light flux =", radiant_flux

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
        print "Small sphere irradiance", irradiance

        radius = 0.1
        area = (4.*pi*radius**2)
        irradiance = radiant_flux / area
        self.light_specs['big_sphere_light'] = """
        larea arealight1 uniform 1. 1. 1. %f
        diffuse black  1 1 1 0.
        s 0 0.8 -0.5 %f
        larea none
        """ % (irradiance, radius)
        print "Big sphere irradiance", irradiance

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
        print "Small cube irradiance", irradiance

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
        print "Big cube irradiance", irradiance

    def buildScenes(self):
        area_light_test_template = """
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
        for name, light_spec in self.light_specs.items():
            self[name] = textwrap.dedent(area_light_test_template.format(**{'light_spec' : light_spec}))


class CornelBoxLightTypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W = 320
        cls.H = 320
        cls.common_opt = { '-w' : cls.W, '-h' : cls.H}
        cls.scenes = CornelBoxLightTypesScenes()

    def runTest(self):
        self.test_forwardtraced()
        self.test_bdpt()

    def test_forwardtraced(self):
        W, H = self.W, self.H
        imagegrid = []
        for name, scene in self.scenes.items():
            image_row = []
            for mode in 'both bsdf lights'.split():
                img = toytrace(
                    scene, '',
                    merge(self.common_opt, { '--pt-sample-mode' : mode, '--max-spp' : 16, }),
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
        for name, scene in self.scenes.items():
            img = toytrace(
                scene, '',
                merge(self.common_opt, {'--algo': 'bdpt', '--max-spp' : 4 }),
                name = name)
            images.append((name, img))
        new_im  = Image.new('RGB', (W, H*len(images)))
        d = ImageDraw.Draw(new_im)
        for i, (name, im) in enumerate(images):
            new_im.paste(im, (0, i*H))
            d.text((1, i*H), name)
        new_im.save(os.path.join(destination_dir, 'arealights_bdpt.png'))


class AtmosphereScenes(SceneCollection):
    def __init__(self):
        SceneCollection.__init__(self, 'atmosphere_')
        scene_template = """
        {camera_spec}

        {light_spec}

        include atmosphere.nff
        """
        light_specs = {
            'zenit' : 'lsun 0 0 -1 100',
            'late' : 'lsun -1 0 -0.02 100',
            'evening' : 'lsun -1 0 -1 100',
            'later' : 'lsun -1 0 0.087 5000  # -5 deg'
        }
        cam_specs = {
            'perspective' : """
        v
        from 0 0 6300.1
        at 1 1 6301
        up 0 0 1
        resolution 640 480
        angle 120
        """,
            'fisheye' : """
        vfisheye
        from 0 0 6300.01
        at 0 0 6301
        up 1 0 0
        resolution 640 640
        """
        }
        for k, v in light_specs.items():
            for l, w in cam_specs.items():
                self[l+'_'+k] = textwrap.dedent(scene_template.format(**{'light_spec' : v, 'camera_spec' : w}))

        spacescene = """
        v
        from -300000 -200000 0
        at 0 0 0
        up 0 1 0
        resolution 640 640
        angle 2.2

        lsun 3 0.7 -0.5 200

        include atmosphere_textured.nff
        """
        backlit = """
        v
        from 0 0 6340
        at -10 0 6339
        up 0 0 1
        resolution 640 640
        angle 20

        lsun 1 0 0.1 200

        include atmosphere_textured.nff
        """
        self['space'] = textwrap.dedent(spacescene)
        self['backlit'] = textwrap.dedent(backlit)


class Atmosphere(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W = 320
        cls.H = 320
        cls.common_opt = {'--max-spp': 16, '-w': cls.W, '-h': cls.H}
        cls.scenes = AtmosphereScenes()

    def test_forwardtraced(self):
        for name, scene in self.scenes.items():
            toytrace(
                scene, 'atmosphere_'+name+'_bdpt.png',
                merge(self.common_opt, {'--algo' : 'bdpt'}),
                name = name)


class VariousScenes(SceneCollection):
    def __init__(self):
        SceneCollection.__init__(self, '')
        self['veach_mis_scene'] = 'veach_mis_scene.nff'
        self['veach_mis_scene_distant'] = 'veach_mis_scene_distant.nff'
        self['material_test_scene'] = 'material_test_scene.nff'
        self['cornel_parallel_arealight'] = 'test_cornel_parallel_arealight.nff'
        self['cornelbox_parabolic_reflector'] = 'test_cornelbox_parabolic_reflector.nff'
        self['env_sphere_mirror'] = 'test_env_sphere_mirror.nff'


class Various(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common_opt = defaultdict(lambda : {'-w': 480, '-h': 320})
        cls.common_opt['cornel_parallel_arealight'] = {'-w': 320, '-h': 320}
        cls.scenes = VariousScenes()

    def runSingle(self, name):
        toytrace(
            self.scenes[name], name + '.png',
            merge(self.common_opt[name], {'--max-spp': 16}),
            name = name)
        toytrace(
            self.scenes[name], name + '_bdpt.png',
            merge(self.common_opt[name], { '--algo' : 'bdpt', '--max-spp': 4}),
            name = name)

    def runTest(self):
        for name, scene in self.scenes.items():
            self.runSingle(name)



if __name__ == '__main__':
    logging.getLogger().setLevel("INFO")
    unittest.main()