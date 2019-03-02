import numpy as np
import matplotlib
import regTrees
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
matplotlib.use('TkAgg')


def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    # See if check box has been selected
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,
                                     regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat,
                                       regTrees.modelTreeEval)
    else:
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0], reDraw.rawDat[:, 1], c="black", s=5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.canvas.draw()


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except Exception:
        tolN = 10
        print("enter Integer for tolN")
        # Clear error and replace with default
        tolNentry.delete(0, tkinter.END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except Exception:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, tkinter.END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)


root = tkinter.Tk()
reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

tkinter.Label(root, text="tolN").grid(row=1, column=0)
tolNentry = tkinter.Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
tkinter.Label(root, text="tolS").grid(row=2, column=0)
tolSentry = tkinter.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
tkinter.Button(root, text="ReDraw",
               command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = tkinter.IntVar()
chkBtn = tkinter.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = regTrees.loadDataSet('sine.txt')
reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]),
                           max(reDraw.rawDat[:, 0]), 0.01)[:, np.newaxis]
reDraw(1.0, 10)

root.mainloop()
