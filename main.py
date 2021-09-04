import tkinter as tk
import tkinter.messagebox
import os


# 定义一个函数功能（内容自己自由编写），供点击Button按键时调用，调用命令参数command=函数名
def open_MC_V():
    tkinter.messagebox.showinfo(title='Note', message='Monte Carlo (state value) will be activated.')
    os.system("python monte_carlo_V.py")

def open_MC_Q():
    tkinter.messagebox.showinfo(title='Note', message='Monte Carlo (state-action value) will be activated.')
    os.system("python monte_carlo_Q.py")


if __name__ == "__main__":
    # 第1步，实例化object，建立窗口window
    window = tk.Tk()

    # 第2步，给窗口的可视化起名字
    window.title('Frozen Lake')

    # 第3步，设定窗口的大小(长 * 宽)
    window.geometry('900x600')  # 这里的乘是小x

    # 第4步，在图形界面上设定标签
    var = tk.StringVar()  # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    l = tk.Label(window, textvariable=var, bg='green', fg='white', font=('Arial', 12), width=30, height=2)
    # 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
    l.pack()

    # 第5步，在窗口界面设置放置Button按键
    mcv_b = tk.Button(window, text='Monte Carlo V', font=('Arial', 12), width=14, height=3, command=open_MC_V)
    mcv_b.pack()
    mcq_b = tk.Button(window, text='Monte Carlo Q', font=('Arial', 12), width=14, height=3, command=open_MC_Q)
    mcq_b.pack()

    # 第6步，主窗口循环显示
    window.mainloop()
