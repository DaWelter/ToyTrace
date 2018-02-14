#!/bin/env/python
# coding: utf-8

import os, sys
import subprocess
from math import pi
import numpy as np
from collections import OrderedDict
import tempfile
import datetime
import hashlib
import textwrap
from PIL import Image, ImageDraw, ImageFont
# References for PIL:
# http://pillow.readthedocs.io/en/5.0.0/reference/ImageDraw.html
# https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python


os.chdir('../scenes')
destination_dir =     os.path.join('..', 'misc', 'renders-'+datetime.datetime.now().strftime('%d-%m-%Y-%H-%M'))
if not os.path.isdir(destination_dir):
    os.mkdir(destination_dir)
scenefile_dir = os.path.join(destination_dir, 'scenes')


def toytrace(scene, output_file, opts):
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


def dumpToSceneDir(scenes, filename_prefix):
    if not os.path.isdir(scenefile_dir):
        os.mkdir(scenefile_dir)
    for k, scene in scenes.items():
        filename = os.path.join(scenefile_dir, filename_prefix+k+'.nff')
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                target_checksum = hashlib.md5(f.read()).digest()
            checksum = hashlib.md5(scene).digest()
            if target_checksum != checksum:
                raise RuntimeError("Attempt to overwrite %s which is different from what would be written!" % filename)
        else:
            with open(filename, 'w') as f:
                f.write(scene)
        scenes[k] = filename


class CornelBoxLightTypes(object):
    def __init__(self):
        self.buildScenePieces()
        self.buildScenes()
        self.items = self.scenes.items

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
        self.scenes = {}
        for name, light_spec in self.light_specs.items():
            self.scenes[name] = textwrap.dedent(area_light_test_template.format(**{'light_spec' : light_spec}))

    def dumpToSceneDir(self):
        dumpToSceneDir(self.scenes, 'cornelbox_')



def runCornelBoxLightTypes():
    W = 320
    H = 320
    imagegrid = []
    scenes = CornelBoxLightTypes()
    for name, scene in scenes.items():
        image_row = []
        for mode in 'both bsdf lights'.split():
            img = toytrace(scene, '', {'--max-spp' : 16, '-w' : W, '-h' : H, '--pt-sample-mode' : mode})
            image_row.append((name, mode, img))
            #print name, mode,'->', img
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


class AtmosphereScenes(object):
    def __init__(self):
        scene_template = """
        {camera_spec}

        {light_spec}

        include atmosphere.nff
        """
        self.scenes = {}
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
                self.scenes[l+'_'+k] = textwrap.dedent(scene_template.format(**{'light_spec' : v, 'camera_spec' : w}))

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
        self.scenes['space'] = textwrap.dedent(spacescene)
        self.scenes['backlit'] = textwrap.dedent(backlit)
        self.items = self.scenes.items

    def dumpToSceneDir(self):
        dumpToSceneDir(self.scenes, 'atmosphere_')


def runAtmosphereScenes():
    W = 320
    H = 320
    for name, scene in AtmosphereScenes().items():
        toytrace(scene, 'atmosphere_'+name+'.png', {'--max-spp' : 16, '-w' : W, '-h' : H })


class ScenesFromFiles(object):
    def __init__(self):
        self.scenes = {
            'veach_mis_scene' : 'veach_mis_scene.nff',
            'veach_mis_scene_distant' : 'veach_mis_scene_distant.nff',
            'material_test_scene' : 'material_test_scene.nff'
        }
        self.items = self.scenes.items


def runScenesFromFiles():
    W = 480
    H = 320
    for name, scene in ScenesFromFiles().items():
        toytrace(scene, name+'.png', {'--max-spp' : 16, '-w' : W, '-h' : H })


if __name__ == '__main__':
    runScenesFromFiles()
    #CornelBoxLightTypes().dumpToSceneDir()
    #runCornelBoxLightTypes()
    #runAtmosphereScenes()
    #AtmosphereScenes().dumpToSceneDir()