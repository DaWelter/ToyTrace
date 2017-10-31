
import os
import sys
import csv
import vtk

inputfile = sys.argv[1]
outputfile = os.path.splitext(inputfile)[0]+'.vtp'

# Following the LongLine example of the VTK documentation.
# https://lorensen.github.io/VTKExamples/site/Python/GeometricObjects/LongLine/
points = vtk.vtkPoints()
lines  = vtk.vtkCellArray()
idx = 0
with open(inputfile, 'r') as f:
  reader = csv.reader(f)
  for row in reader:
    x1, x2, x3, y1, y2, y3, t = row
    x = map(float, [x1, x2, x3])
    y = map(float, [y1, y2, y3])
    points.InsertNextPoint(x)
    points.InsertNextPoint(y)
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, idx)
    line.GetPointIds().SetId(1, idx+1)
    lines.InsertNextCell(line)
    idx += 2

polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetLines(lines)
polydata.Modified()
polydata.Update()

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(outputfile)
writer.SetInput(polydata)
writer.Write()