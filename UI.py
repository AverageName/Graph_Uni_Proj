from tkinter import *
from tkinter.ttk import Combobox, Progressbar
from tkinter import messagebox
from PIL import ImageTk, Image
from functools import partial
from MetricsCalculator import MetricsCalculator
import time
import sys
from utils.utils import find_min_path


class App:
    def __init__(self, mc):
        self.mc = mc
        self.window = Tk()
        self.window.title("Екатеринбург")
        self.window.geometry("1800x1000")
        self.result = ''
        self.n_1 = self.m_1 = None
        self.modes = ('fwd', 'bwd', 'fwd_bwd')

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
        Label(self.window, text='M = ').grid(column=0, row=4, sticky=E)
        self.m_1_input = Spinbox(self.window, width=5, from_=1, to=10)
        self.m_1_input.grid(column=1, row=4, sticky=W)
        Label(self.window, text='N = ').grid(column=0, row=5, sticky=E)
        self.n_1_input = Spinbox(self.window, width=5, from_=1, to=100)
        self.n_1_input.grid(column=1, row=5, sticky=W)

        Button(self.window, text='Выбрать', command=self.set_m_n_1).grid(column=1, row=6, sticky=W, pady=1)
        self.show_chosen_1 = Button(self.window, text='Показать выбранные',
                                    command=partial(self.set_image, "points_for_1_part"))
        Label(self.window, text='mode(1): ').grid(column=0, row=8, sticky=E)
        self.mode_1 = Combobox(self.window, width=15)
        self.mode_1.grid(column=1, row=8, sticky=W)
        self.mode_1['values'] = ("Туда", "Обратно", "Туда-обратно")
        self.mode_1.current(0)
        self.task_1_1_a = Button(self.window, text='Рассчитать 1а', command=self.count_first_a)
        self.show_1_1_a = Button(self.window, text='Show', command=partial(self.set_image, 'task_1_a'))
        Label(self.window, text='X = ').grid(column=0, row=9, sticky=E)
        self.x = Entry(self.window, width=7)
        self.x.grid(column=1, row=9, sticky=W)
        self.task_1_1_b = Button(self.window, text='1б', command=self.count_first_b)
        self.show_1_1_b = Button(self.window, text='Show', command=partial(self.set_image, 'task_1_b'))

        Label(self.window, text='mode(2): ').grid(column=0, row=10, sticky=E)
        self.mode_2 = Combobox(self.window, width=15)
        self.mode_2.grid(column=1, row=10, sticky=W)
        self.mode_2['values'] = ("Туда", "Обратно", "Туда-обратно")
        self.mode_2.current(0)
        self.task_1_2 = Button(self.window, text='п. 2', command=self.count_second)
        self.show_1_2 = Button(self.window, text='Show', command=partial(self.set_image, 'task_2'))
        self.task_1_3 = Button(self.window, text='п. 3', command=self.count_third)
        self.show_1_3 = Button(self.window, text='Show', command=partial(self.set_image, 'task_3'))
        self.task_1_4 = Button(self.window, text='п. 4', command=self.count_fourth)
        self.show_1_4 = Button(self.window, text='Show', command=partial(self.set_image, 'task_4'))
        Label(self.window, text='', height=1).grid(column=1, row=13)
        self.wait_first = Label(self.window, text='', fg='red')
        self.wait_first.grid(column=1, row=14)

        Label(self.window, text='', height=2).grid(column=1, row=15)
        Label(self.window, text='Часть 2').grid(column=1, row=16)
        Button(self.window, text='Показать карту инфраструктурных объектов',
               command=partial(self.set_image, "inf_objs")).grid(column=1, row=17, pady=2)
        Label(self.window, text='N = ').grid(column=1, row=18)
        self.n_input = Spinbox(self.window, from_=1, to=100, width=5)
        self.n_input.grid(column=1, row=18, sticky=E)
        Label(self.window, text='Индекс инф. объекта: ').grid(column=1, row=19)
        self.inf_index = Spinbox(self.window, from_=0, to=21, width=5)
        self.inf_index.grid(column=1, row=19, sticky=E)
        Button(self.window, text='Рассчитать', command=self.count_sp).grid(column=1, row=20)

        Label(self.window, text='', height=2).grid(column=1, row=21)
        self.progress = Progressbar(self.window, orient=HORIZONTAL, length=0, mode='determinate')
        self.progress.grid(column=1, row=22)
        self.progress_info = Label(self.window, text='')
        self.progress_info.grid(column=1, row=23)

        Label(self.window, text='', height=2).grid(column=1, row=24)
        self.combo = Combobox(self.window, width=30)
        self.combo['values'] = (
            "Дерево кратчайших путей", "Дендрограмма",
            "Разделение на 2 кластера", "Деревья 2 центроид", "Дерево от 2 центроид до объекта",
            "Разделение на 3 кластера", "Деревья 3 центроид", "Дерево от 3 центроид до объекта",
            "Разделение на 5 кластеров", "Деревья 5 центроид", "Дерево от 5 центроид до объекта"
        )
        self.combo.current(0)
        self.show_btn = Button(self.window, text='Показать', command=self.set_combo_image)
        self.add_recount = Button(self.window, text='Добавить пункт назначения и пересчитать',
                                  command=self.add_and_recount)

        Label(self.window, text='Test').grid(column=1, row=29, pady=3, sticky=W)
        Label(self.window, text='File:').grid(column=0, row=30, pady=1, sticky=E)
        self.filename = Entry(self.window, width=15)
        self.filename.grid(column=1, row=30, sticky=W)
        Label(self.window, text='Start:').grid(column=0, row=31, pady=1, sticky=E)
        self.start_id = Entry(self.window, width=5)
        self.start_id.grid(column=1, row=31, sticky=W)
        Label(self.window, text='Dest:').grid(column=0, row=32, pady=1, sticky=E)
        self.dest_id = Entry(self.window, width=5)
        self.dest_id.grid(column=1, row=32, sticky=W)
        self.count_test = Button(self.window, text='count', command=self.count_test)
        self.count_test.grid(column=1, row=33, pady=2, sticky=W)

        Button(self.window, text='exit', command=partial(sys.exit, 0)).grid(column=1, row=34, pady=5)

    def count_test(self):
        try:
            start = int(self.start_id.get())
            dest = int(self.dest_id.get())
            res = find_min_path(self.filename.get(), start, dest)
            messagebox.showinfo('result', 'length: {}\npath: {}'.format(res[0], res[1]))
            self.window.mainloop()
        except ValueError:
            messagebox.showinfo('m n', 'M|N should be numeric')

    def toggle_waiting(self, on):
        if on:
            self.wait_first.configure(text='wait please...')
        else:
            self.wait_first.configure(text='')
        self.window.update()

    def set_m_n_1(self):
        try:
            n_1 = int(self.n_1_input.get())
            m_1 = int(self.m_1_input.get())
            self.toggle_waiting(True)
            self.hide_info()
            self.mc.set_objs(n_1, m=m_1)
            self.set_image('points_for_1_part')
            self.show_first_part()
            self.toggle_waiting(False)
            self.window.mainloop()
        except ValueError:
            messagebox.showinfo('m n', 'M|N should be numeric')

    def show_first_part(self):
        self.show_chosen_1.grid(column=1, row=7, pady=1)
        self.task_1_1_a.grid(column=1, row=8, sticky=E)
        self.task_1_1_b.grid(column=1, row=9, sticky=E)
        self.task_1_2.grid(column=1, row=10, sticky=E)
        self.task_1_3.grid(column=1, row=11, pady=3, sticky=E)
        self.task_1_4.grid(column=1, row=12, sticky=E)

    def hide_first_part(self):
        self.show_chosen_1.grid_forget()
        self.task_1_1_a.grid_forget()
        self.task_1_1_b.grid_forget()
        self.task_1_2.grid_forget()
        self.task_1_3.grid_forget()
        self.task_1_4.grid_forget()
        self.show_1_1_a.grid_forget()
        self.show_1_1_b.grid_forget()
        self.show_1_2.grid_forget()
        self.show_1_3.grid_forget()
        self.show_1_4.grid_forget()

    def count_first_a(self):
        self.toggle_waiting(True)
        index = self.mode_1['values'].index(self.mode_1.get())
        res = self.mc.nearest(self.modes[index])
        self.set_image('task_1_a')
        self.show_1_1_a.grid(column=2, row=8, padx=1, sticky=W)
        self.toggle_waiting(False)
        messagebox.showinfo('1a', res)
        self.window.mainloop()

    def count_first_b(self):
        index = self.mode_1['values'].index(self.mode_1.get())
        try:
            x = int(self.x.get())
            self.toggle_waiting(True)
            res = self.mc.closer_than_x(x, self.modes[index])
            self.set_image('task_1_b')
            self.show_1_1_b.grid(column=2, row=9, padx=1, sticky=W)
            self.toggle_waiting(False)
            messagebox.showinfo('1б', res)
            self.window.mainloop()
        except ValueError:
            messagebox.showinfo('error', 'x should be numeric')
            return

    def count_second(self):
        self.toggle_waiting(True)
        index = self.mode_2['values'].index(self.mode_2.get())
        res = self.mc.min_furthest_for_inf(self.modes[index])
        self.set_image('task_2')
        self.show_1_2.grid(column=2, row=10, padx=1, sticky=W)
        self.toggle_waiting(False)
        messagebox.showinfo('2', res)
        self.window.mainloop()

    def count_third(self):
        self.toggle_waiting(True)
        res = self.mc.closest_inf_in_summary()
        self.set_image('task_3')
        self.show_1_3.grid(column=2, row=11, padx=1, sticky=W)
        self.toggle_waiting(False)
        messagebox.showinfo('3', res)
        self.window.mainloop()

    def count_fourth(self):
        self.toggle_waiting(True)
        res = self.mc.min_weight_tree()
        self.set_image('task_4')
        self.show_1_4.grid(column=2, row=12, padx=1, sticky=W)
        self.toggle_waiting(False)
        messagebox.showinfo('4', res)
        self.window.mainloop()

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
        self.hide_first_part()
        n = int(self.n_input.get())
        inf_obj_i = int(self.inf_index.get())
        self.mc.set_objs(n)
        self.mc.set_inf_obj(inf_obj_i)
        self.second_part()

    def add_and_recount(self):
        self.mc.add_obj()
        self.n_input.selection_adjust(5)
        self.second_part()

    def second_part(self):
        self.progress.configure(length=200)
        self.hide_info()
        time_wt_plot = 0
        start = time.time()
        self.update_progress(0, 'saving to csv...')
        self.mc.save_chosen_objs_to_csv()
        self.update_progress(11, 'counting tree...')
        sum_, weight, routes_list, o_w_r, t = self.mc.list_to_obj_tree(self.mc.chosen_objs, self.mc.chosen_inf_obj,
                                                                   filename='./csv/min_tree.csv')
        time_wt_plot += t
        # messagebox.showinfo('1', 'sum: {}\nweight: {}'.format(sum_, weight))
        self.update_progress(22, 'saving tree...')
        self.mc.save_tree_plot(routes_list, [self.mc.graph.nodes[routes_list[0][0]]], 'routes_to_random_inf', o_w_r)
        self.set_image('routes_to_random_inf')

        self.update_progress(33, 'researching clusters...')
        clusters, history, t = self.mc.objs_into_clusters(1, write=True)
        time_wt_plot += t
        self.update_progress(44, 'creating dendrogram...')
        self.mc.dendrogram(clusters, history)
        self.set_image('dendrogram')

        self.update_progress(55, 'working with 2 clusters...')
        cs_2, w_2, cs_2_c, w_2_c, t = self.mc.work_with_clusters(history, 2)
        time_wt_plot += t
        self.set_image('2_clusters')

        self.update_progress(70, 'working with 3 clusters...')
        cs_3, w_3, cs_3_c, w_3_c, t = self.mc.work_with_clusters(history, 3)
        time_wt_plot += t
        self.set_image('3_clusters')

        self.update_progress(85, 'working with 5 clusters...')
        cs_5, w_5, cs_5_c, w_5_c, t = self.mc.work_with_clusters(history, 5)
        time_wt_plot += t
        self.set_image('5_clusters')
        self.update_progress(100, 'done!')

        result = "min weight tree: \n\tsum: {}\n\tweight {}\n" \
                 "2 centroids tree: \n\tsum: {}\n\tweight {}\n\tcentroid_sum: {}\n\tcentroid_weight: {}\n" \
                 "3 centroids tree: \n\tsum: {}\n\tweight {}\n\tcentroid_sum: {}\n\tcentroid_weight: {}\n" \
                 "5 centroids tree: \n\tsum: {}\n\tweight {}\n\tcentroid_sum: {}\n\tcentroid_weight: {}\n" \
                 "time without plotting: {}\ntime: {}"\
            .format(sum_, weight,
                    cs_2, w_2, cs_2_c, w_2_c,
                    cs_3, w_3, cs_3_c, w_3_c,
                    cs_5, w_5, cs_5_c, w_5_c,
                    time_wt_plot, time.time() - start)
        self.set_image('routes_to_random_inf')
        messagebox.showinfo('result', result)

        self.show_info()
        self.window.mainloop()

    def update_progress(self, val, info):
        self.progress['value'] = val
        self.progress_info.configure(text=info)
        self.window.update()

    def hide_info(self):
        self.progress['value'] = 0
        self.combo.grid_forget()
        self.show_btn.grid_forget()
        self.add_recount.grid_forget()
        self.progress_info.configure(text='')
        self.window.update()

    def show_info(self):
        self.combo.grid(column=1, row=26)
        self.show_btn.grid(column=1, row=27)
        self.add_recount.grid(column=1, row=28, pady=5)
        self.window.update()

    def start_loop(self):
        self.window.mainloop()


m = MetricsCalculator('./Ekb.osm')
m.crop_and_save_graph()

app = App(m)
app.start_loop()




# set_ = set()
# set_.add((1, 2))
# set_.add((2, 3))
# set_.add((1, 2))
#
# if (2, 3) not in set_:
#     print('no')
# print(set_)