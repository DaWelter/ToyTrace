import path_guiding_kdtree_loader

import vtk
import sys
import os
import json
from pprint import pprint, pformat
import numpy as np
import itertools
from vtk.util import numpy_support


if __name__ == '__main__':

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.SetWindowName("Cube")
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(ren)
    iren.SetInteractorStyle(style)
    colors = vtk.vtkNamedColors()


    with open('/tmp/paths.json') as f:
        data = f.read()
    paths = map(str.strip, filter(lambda s: s, data.split("\n-----------------------------------------\n")))
    paths = list(map(lambda s: json.loads(s), paths))
    for p in paths:
        p['max_contribution'] = max(*p['contribution']['pixel_value'])

    paths = sorted(paths, key=lambda p: p['max_contribution'])
    paths = paths[-10:]

    points = vtk.vtkPoints()
    lines  = vtk.vtkCellArray()
    idx = 0
    for p in paths:
        nodes = p['nodes']
        for i in range(len(nodes)-1):
            points.InsertNextPoint(nodes[i]['position'])
            points.InsertNextPoint(nodes[i+1]['position'])
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, idx)
            line.GetPointIds().SetId(1, idx+1)
            lines.InsertNextCell(line)
            idx += 2

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.Modified()
    #polydata.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    ren.AddActor(actor)

    if os.path.isfile("/tmp/scene.obj") and 1:
        # Because of the number format
        #os.environ['LC_NUMERIC']='en_US.UTF-8'
        # Not working though. Must change this in the terminal :-(
        importer = vtk.vtkOBJReader()
        importer.SetFileName("/tmp/scene.obj")
        importer.Update()
        pd = importer.GetOutput()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.5)
        ren.AddActor(actor)

    axes = vtk.vtkAxesActor()
    ren.AddActor(axes)

    ren.ResetCamera()
    ren.GetActiveCamera().Azimuth(30)
    ren.GetActiveCamera().Elevation(30)
    ren.ResetCameraClippingRange()
    #ren.GetActiveCamera().SetPosition(-1, 1, -1)
    ren.SetBackground(colors.GetColor3d("Black"))

    # Enable user interface interactor.
    iren.Initialize()
    renWin.Render()
    iren.Start()