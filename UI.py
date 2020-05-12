from tkinter import *
from tkinter.ttk import Combobox
from tkinter import messagebox
from PIL import ImageTk, Image
from functools import partial
from MetricsCalculator import MetricsCalculator


class App:
    def __init__(self, mc):
        self.mc = mc
        self.window = Tk()
        self.window.title("Екатеринбург")
        self.window.geometry("1800x1000")
        self.result = ''

        self.load = Image.open("./images/Ekb_graph_cropped.png")
        self.image = ImageTk.PhotoImage(self.load)
        self.img = Label(self.window, image=self.image)
        self.img.grid(column=3, columnspan=100, row=1, rowspan=100)

        Label(self.window, text='').grid(row=0)
        Label(self.window, text='', width=10).grid(column=0)
        Label(self.window, width=10).grid(column=2)

        Button(self.window, text='Показать Екатеринбург',
               command=partial(self.set_image, "Ekb_graph_cropped")).grid(column=1, row=1)
        Label(self.window, text='').grid(column=1, row=2)

        Label(self.window, text='Часть 1').grid(column=1, row=3)
        Button(self.window, text='Рассчитать 1.а,б').grid(column=1, row=4)
        Label(self.window, text='Вывод в файле ...').grid(column=1, row=5)
        Button(self.window, text='Показать объект из п.2').grid(column=1, row=6)
        Button(self.window, text='Показать объект из п.3').grid(column=1, row=7)
        Button(self.window, text='Показать объект из п.4').grid(column=1, row=8)

        Label(self.window, text='', height=4).grid(column=1, row=9)
        Label(self.window, text='Часть 2').grid(column=1, row=10)
        Button(self.window, text='Показать карту инфраструктурных объектов',
               command=partial(self.set_image, "inf_objs")).grid(column=1, row=11)
        Label(self.window, text='N = ').grid(column=0, row=12, sticky=E)
        self.n_input = Spinbox(self.window, from_=1, to=100, width=5)
        self.n_input.grid(column=1, row=12, sticky=W)
        Label(self.window, text='Индекс инф. объекта: ').grid(column=0, row=13, sticky=E)
        self.inf_index = Spinbox(self.window, from_=0, to=21, width=5)
        self.inf_index.grid(column=1, row=13, sticky=W)
        Button(self.window, text='Рассчитать', command=self.count_second_part).grid(column=1, row=14)

        Label(self.window, text='', height=3).grid(column=1, row=15)
        self.combo = Combobox(self.window, width=30)
        self.combo['values'] = (
            "Дерево кратчайших путей", "Дендрограмма",
            "Разделение на 2 кластера", "Деревья 2 центроид", "Дерево от 2 центроид до объекта",
            "Разделение на 3 кластера", "Деревья 3 центроид", "Дерево от 3 центроид до объекта",
            "Разделение на 5 кластеров", "Деревья 5 центроид", "Дерево от 5 центроид до объекта"
        )
        self.combo.current(0)
        self.combo.grid(column=1, row=16)
        Button(self.window, text='Показать', command=self.set_combo_image).grid(column=1, row=17)

        Label(self.window, text='').grid(column=1, row=18)
        Button(self.window, text='Добавить пункт назначения и пересчитать').grid(column=1, row=19)

    def set_image(self, name):
        self.load = Image.open("./images/{}.png".format(name))
        self.image = ImageTk.PhotoImage(self.load)
        self.img.configure(image=self.image)

    def set_combo_image(self):
        img_names = (
            "routes_to_random_inf", "dendrogram",
            "2_clusters", "2_clusters_trees", "2_centroids_tree",
            "3_clusters", "3_clusters_trees", "3_centroids_tree",
            "5_clusters", "5_clusters_trees", "5_centroids_tree",
        )
        index = self.combo['values'].index(self.combo.get())
        self.set_image(img_names[index])

    def count_second_part(self):
        n = int(self.n_input.get())
        inf_obj_i = int(self.inf_index.get())
        self.mc.set_objs(n)
        self.mc.set_inf_obj(inf_obj_i)
        cs_2, cs_3, cs_5 = self.mc.second_part()
        self.result = "2 centroids tree: {}\n3 centroids tree: {}\n5 centroids tree: {}".format(cs_2, cs_3, cs_5)
        self.set_image('routes_to_random_inf')
        messagebox.showinfo('result', self.result)

    def start_loop(self):
        self.window.mainloop()


m = MetricsCalculator('./Ekb.osm')
m.crop_and_save_graph()

app = App(m)
app.start_loop()
