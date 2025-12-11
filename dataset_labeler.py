import os
from tkinter import *
from PIL import Image, ImageTk

FOLDER = "cool_duck_images"
EXTS = (".jpg", ".jpeg", ".png", ".gif")
MAX_W, MAX_H = 1800, 1200

class ImageBrowser:
    def __init__(self, root):
        self.root = root
        self.files = [f for f in os.listdir(FOLDER) if f.lower().endswith(EXTS)]
        self.index = 0
        self.points = {f: [] for f in self.files}
        self.preview = None
        self.overlay_ids = []

        self.canvas = Canvas(root, bg="black")
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Motion>", self.hover)
        self.canvas.bind("<Leave>", lambda e: self.clear_preview())
        
        root.bind("z", lambda e: self.undo())

        btns = Frame(root); btns.pack()
        Button(btns, text="Prev", command=self.prev).pack(side=LEFT)
        Button(btns, text="Next", command=self.next).pack(side=LEFT)
        Button(btns, text="Undo", command=self.undo).pack(side=LEFT)
        Button(btns, text="Export", command=self.export).pack(side=LEFT)

        self.show()

    def resize_aspect(self, img):
        w, h = img.size
        s = min(MAX_W/w, MAX_H/h)
        return img.resize((int(w*s), int(h*s)), Image.LANCZOS), s

    def scaled_to_original(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def original_to_scaled(self, x, y):
        return int(x * self.scale), int(y * self.scale)

    def clear_overlay(self):
        for oid in self.overlay_ids:
            self.canvas.delete(oid)
        self.overlay_ids = []

    def draw_overlay(self):
        self.clear_overlay()
        pts = self.points[self.files[self.index]]
        scaled_pts = [self.original_to_scaled(x, y) for x, y in pts]

        # preview
        if self.preview and len(pts) < 4:
            scaled_pts_preview = scaled_pts + [self.preview]
        else:
            scaled_pts_preview = scaled_pts

        # draw points
        for x, y in scaled_pts_preview:
            r = 4
            oid = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="red", outline="")
            self.overlay_ids.append(oid)

        # draw lines A-B-C-D-A
        if len(scaled_pts_preview) >= 2:
            for i in range(1, len(scaled_pts_preview)):
                x1, y1 = scaled_pts_preview[i-1]
                x2, y2 = scaled_pts_preview[i]
                oid = self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2)
                self.overlay_ids.append(oid)

        if len(scaled_pts_preview) == 4:
            x1, y1 = scaled_pts_preview[0]
            x2, y2 = scaled_pts_preview[3]
            oid = self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2)
            self.overlay_ids.append(oid)

    def show(self):
        name = self.files[self.index]

        # prepare background image once
        self.img_orig = Image.open(os.path.join(FOLDER, name))
        self.img, self.scale = self.resize_aspect(self.img_orig.copy())
        self.tk = ImageTk.PhotoImage(self.img)

        self.canvas.config(width=self.img.width, height=self.img.height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk)

        self.draw_overlay()

    def add_point(self, e):
        name = self.files[self.index]
        if len(self.points[name]) < 4:
            ox, oy = self.scaled_to_original(e.x, e.y)
            self.points[name].append((ox, oy))
        self.preview = None
        self.draw_overlay()

    def hover(self, e):
        name = self.files[self.index]
        if len(self.points[name]) < 4:
            self.preview = (e.x, e.y)
            self.draw_overlay()

    def clear_preview(self):
        self.preview = None
        self.draw_overlay()

    def undo(self):
        name = self.files[self.index]
        if self.points[name]:
            self.points[name].pop()
        self.preview = None
        self.draw_overlay()

    def next(self):
        self.index = (self.index + 1) % len(self.files)
        self.preview = None
        self.show()

    def prev(self):
        self.index = (self.index - 1) % len(self.files)
        self.preview = None
        self.show()

    def export(self):
        for name, pts in self.points.items():
            line = name + ": " + ", ".join(f"({x}, {y})" for x, y in pts)
            print(line)

root = Tk()
ImageBrowser(root)
root.mainloop()

