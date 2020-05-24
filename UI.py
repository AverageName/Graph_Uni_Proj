from tkinter import *
from tkinter.ttk import Combobox, Progressbar
from tkinter import messagebox
from PIL import ImageTk, Image
from functools import partial
from MetricsCalculator import MetricsCalculator
import time


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
               command=partial(self.set_image, "Ekb_graph_cropped")).grid(column=1, row=1, pady=3)
        Label(self.window, text='').grid(column=1, row=2)

        Label(self.window, text='Часть 1').grid(column=1, row=3)
        Label(self.window, text='start: ').grid(column=0, row=3, sticky=E)
        self.start_1 = Combobox(self.window, width=10)
        self.start_1.grid(column=1, row=3, sticky=W)
        self.start_1['values'] = ("inf", "node")
        self.start_1.current(0)
        Label(self.window, text='X = ').grid(column=0, row=4, sticky=E)
        self.x = Entry(self.window, width=10)
        self.x.grid(column=1, row=4, sticky=W)
        Button(self.window, text='Рассчитать 1.а,б', command=self.count_first).grid(column=1, row=4, sticky=E)
        self.csv_name = Label(self.window, text='')
        self.csv_name.grid(column=1, row=5, sticky=E)
        Label(self.window, text='mode(1): ').grid(column=0, row=5, sticky=E)
        self.mode_1 = Combobox(self.window, width=10)
        self.mode_1.grid(column=1, row=5, sticky=W)
        self.mode_1['values'] = ("Туда", "Обратно", "Туда-обратно")
        self.mode_1.current(0)

        Label(self.window, text='mode(2): ').grid(column=0, row=6, sticky=E)
        self.mode_2 = Combobox(self.window, width=10)
        self.mode_2.grid(column=1, row=6, sticky=W)
        self.mode_2['values'] = ("Туда", "Обратно", "Туда-обратно")
        self.mode_2.current(0)
        Button(self.window, text='Показать объект из п.2', command=self.count_second).grid(column=1, row=6, sticky=E)
        Button(self.window, text='Показать объект из п.3', command=self.count_third)\
            .grid(column=1, row=7, pady=3, sticky=E)
        Button(self.window, text='Показать объект из п.4', command=self.count_fourth).grid(column=1, row=8, sticky=E)

        Label(self.window, text='', height=4).grid(column=1, row=9)
        Label(self.window, text='Часть 2').grid(column=1, row=10)
        Button(self.window, text='Показать карту инфраструктурных объектов',
               command=partial(self.set_image, "inf_objs")).grid(column=1, row=11, pady=2)
        Label(self.window, text='N = ').grid(column=1, row=12)
        self.n_input = Spinbox(self.window, from_=1, to=100, width=5)
        self.n_input.grid(column=1, row=12, sticky=E)
        Label(self.window, text='Индекс инф. объекта: ').grid(column=1, row=13)
        self.inf_index = Spinbox(self.window, from_=0, to=21, width=5)
        self.inf_index.grid(column=1, row=13, sticky=E)
        Button(self.window, text='Рассчитать', command=self.count_sp).grid(column=1, row=14)

        Label(self.window, text='', height=2).grid(column=1, row=15)
        self.progress = Progressbar(self.window, orient=HORIZONTAL, length=0, mode='determinate')
        self.progress.grid(column=1, row=16)

        Label(self.window, text='', height=2).grid(column=1, row=17)
        self.combo = Combobox(self.window, width=30)
        self.combo['values'] = (
            "Дерево кратчайших путей", "Дендрограмма",
            "Разделение на 2 кластера", "Деревья 2 центроид", "Дерево от 2 центроид до объекта",
            "Разделение на 3 кластера", "Деревья 3 центроид", "Дерево от 3 центроид до объекта",
            "Разделение на 5 кластеров", "Деревья 5 центроид", "Дерево от 5 центроид до объекта"
        )
        self.combo.current(0)
        self.show_btn = Button(self.window, text='Показать', command=self.set_combo_image)

        Label(self.window, text='').grid(column=1, row=20)
        self.add_recount = Button(self.window, text='Добавить пункт назначения и пересчитать',
                                  command=self.add_and_recount)

    def count_first(self):
        self.csv_name.configure(text='')
        try:
            x = int(self.x.get())
            time.sleep(3)
            self.csv_name.configure(text='some_file.csv')
        except ValueError:
            messagebox.showinfo('error', 'x should be numeric')
            return

    def count_second(self):
        modes = ('fwd', 'bwd', 'fwd_bwd')
        index = self.mode_2['values'].index(self.mode_2.get())
        messagebox.showinfo('mode', modes[index])

    def count_third(self):
        messagebox.showinfo('3', 'hello')

    def count_fourth(self):
        messagebox.showinfo('4', 'hello')

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

    def count_sp(self):
        n = int(self.n_input.get())
        inf_obj_i = int(self.inf_index.get())
        self.mc.set_objs(n, 0)
        self.mc.set_inf_obj(inf_obj_i)
        self.second_part()

    def add_and_recount(self):
        self.mc.add_obj()
        self.n_input.selection_adjust(5)
        self.second_part()

    def second_part(self):
        self.progress.configure(length=200)
        self.hide_info()

        start = time.time()
        self.mc.save_chosen_objs_to_csv()
        self.update_progress(11)
        sum_, routes_list, o_w_r = self.mc.list_to_obj_tree(self.mc.chosen_objs, self.mc.chosen_inf_obj,
                                                     filename='./csv/min_tree.csv')
        self.update_progress(22)
        self.mc.save_tree_plot(routes_list, [self.mc.graph.nodes[routes_list[0][0]]], 'routes_to_random_inf', o_w_r)
        self.update_progress(33)
        clusters, history = self.mc.objs_into_clusters(1, write=True)
        self.update_progress(44)
        self.mc.dendrogram(clusters, history)
        self.update_progress(55)
        cs_2 = self.mc.work_with_clusters(history, 2)
        self.update_progress(70)
        cs_3 = self.mc.work_with_clusters(history, 3)
        self.update_progress(85)
        cs_5 = self.mc.work_with_clusters(history, 5)
        self.update_progress(100)

        result = "2 centroids tree: {}\n3 centroids tree: {}\n5 centroids tree: {}\ntime: {}"\
            .format(cs_2, cs_3, cs_5, time.time() - start)
        self.set_image('routes_to_random_inf')
        messagebox.showinfo('result', result)

        self.show_info()
        self.window.mainloop()

    def update_progress(self, val):
        self.progress['value'] = val
        self.window.update()

    def hide_info(self):
        self.progress['value'] = 0
        self.combo.grid_forget()
        self.show_btn.grid_forget()
        self.add_recount.grid_forget()
        self.window.update()

    def show_info(self):
        self.combo.grid(column=1, row=18)
        self.show_btn.grid(column=1, row=19)
        self.add_recount.grid(column=1, row=21)
        self.window.update()

    def start_loop(self):
        self.window.mainloop()


m = MetricsCalculator('./Ekb.osm')
m.crop_and_save_graph()

app = App(m)
app.start_loop()
