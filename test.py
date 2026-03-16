import docx
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


def create_formatted_doc():
    doc = docx.Document()

    # 设置基础字体
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)

    # 标题
    title = doc.add_heading('一种自适应惯性权重与精英局部搜索的改进粒子群算法及其仿真研究', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 摘要
    doc.add_heading('摘要', level=1)
    p = doc.add_paragraph()
    p.add_run(
        '粒子群优化（PSO）因参数少、实现简单而被广泛用于连续优化问题，但其在复杂多峰函数上易出现收敛停滞与早熟现象。为提升全局搜索能力与后期精细寻优能力，本文提出一种改进粒子群算法：在标准PSO框架下引入线性递减惯性权重，根据粒子成功更新比例自适应调整学习因子，并对全局最优粒子实施概率递增的精英局部搜索（高斯扰动），其步长与群体多样性相关。在Sphere、Rosenbrock、Rastrigin与Griewank四个基准函数（维度30）上进行仿真，对比30次独立运行的统计结果与平均收敛曲线。实验表明，所提方法在多数测试函数上获得更低的目标函数值，并表现出更快或更稳定的收敛趋势。')

    p = doc.add_paragraph()
    p.add_run('关键词：').bold = True
    p.add_run(' 粒子群优化；自适应参数；局部搜索；群体智能；函数优化')

    # Abstract
    doc.add_heading('Abstract', level=1)
    p = doc.add_paragraph(
        'Particle Swarm Optimization (PSO) is widely applied to continuous optimization problems due to its few parameters and simple implementation. However, it tends to suffer from convergence stagnation and premature convergence when tackling complex multimodal functions. To enhance its global search capability and fine-grained optimization performance in the late iteration stage, this paper proposes an improved PSO algorithm. Within the framework of standard PSO, the proposed algorithm introduces a linearly decreasing inertia weight, adaptively adjusts the learning factors according to the proportion of particles with successful updates, and performs an elitist local search (Gaussian perturbation) with a probability-increasing strategy on the globally optimal particle, where the perturbation step size is correlated with population diversity. Simulation experiments were conducted on four benchmark functions (Sphere, Rosenbrock, Rastrigin, and Griewank) with a dimension of 30. Experimental results demonstrate that the proposed method achieves lower objective function values on most test functions.')

    p = doc.add_paragraph()
    p.add_run('Keywords: ').bold = True
    p.add_run(
        'Particle Swarm Optimization, Adaptive Parameters, Local Search, Swarm Intelligence, Function Optimization')

    # 1. 引言
    doc.add_heading('1. 引言', level=1)
    doc.add_paragraph(
        '智能优化算法通常以群体协作与随机搜索为核心思想，通过在搜索空间中并行探索与信息共享来逼近全局最优解。粒子群优化（Particle Swarm Optimization, PSO）模拟鸟群/鱼群觅食行为，具有实现简单、收敛速度快等优点。然而在高维或多峰问题中，标准PSO常因群体多样性降低而陷入局部最优，后期改进空间有限。因此，如何在保持PSO简洁性的同时增强其探索-开发（exploration-exploitation）平衡，是值得研究的方向。')

    # 2. 标准PSO算法
    doc.add_heading('2. 标准PSO算法', level=1)
    p = doc.add_paragraph('在D维连续空间中，第i个粒子在迭代t时刻的位置与速度分别为 ')
    p.add_run('x').italic = True
    p.add_run('i').font.subscript = True
    p.add_run('t').font.superscript = True
    p.add_run(' ∈ R')
    p.add_run('D').font.superscript = True
    p.add_run(' 与 ')
    p.add_run('v').italic = True
    p.add_run('i').font.subscript = True
    p.add_run('t').font.superscript = True
    p.add_run(' ∈ R')
    p.add_run('D').font.superscript = True
    p.add_run('。粒子记忆自身历史最优位置 ')
    p.add_run('p').italic = True
    p.add_run('i').font.subscript = True
    p.add_run('（个体最优），群体共享历史最优位置 ')
    p.add_run('g').italic = True
    p.add_run('（全局最优）。')

    doc.add_heading('2.1 速度与位置更新', level=2)
    doc.add_paragraph('标准PSO的更新公式如下：')

    # Formula 1
    p_eq = doc.add_paragraph()
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p_eq.add_run('v')
    run.italic = True
    run = p_eq.add_run('i')
    run.font.subscript = True
    run = p_eq.add_run('t+1')
    run.font.superscript = True
    p_eq.add_run(' = ω·')
    run = p_eq.add_run('v')
    run.italic = True
    run = p_eq.add_run('i')
    run.font.subscript = True
    run = p_eq.add_run('t')
    run.font.superscript = True
    p_eq.add_run(' + c')
    p_eq.add_run('1').font.subscript = True
    p_eq.add_run('·r')
    p_eq.add_run('1').font.subscript = True
    p_eq.add_run('·(')
    p_eq.add_run('p').italic = True
    p_eq.add_run('i').font.subscript = True
    p_eq.add_run(' - ')
    p_eq.add_run('x').italic = True
    p_eq.add_run('i').font.subscript = True
    p_eq.add_run('t')
    p_eq.add_run(')')

    p_eq.add_run(' + c')
    p_eq.add_run('2').font.subscript = True
    p_eq.add_run('·r')
    p_eq.add_run('2').font.subscript = True
    p_eq.add_run('·(')
    p_eq.add_run('g').italic = True
    p_eq.add_run(' - ')
    p_eq.add_run('x').italic = True
    p_eq.add_run('i').font.subscript = True
    p_eq.add_run('t')
    p_eq.add_run(')')

    # Formula 2
    p_eq2 = doc.add_paragraph()
    p_eq2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p_eq2.add_run('x')
    run.italic = True
    run = p_eq2.add_run('i')
    run.font.subscript = True
    run = p_eq2.add_run('t+1')
    run.font.superscript = True
    p_eq2.add_run(' = ')
    run = p_eq2.add_run('x')
    run.italic = True
    run = p_eq2.add_run('i')
    run.font.subscript = True
    run = p_eq2.add_run('t')
    run.font.superscript = True
    p_eq2.add_run(' + ')
    run = p_eq2.add_run('v')
    run.italic = True
    run = p_eq2.add_run('i')
    run.font.subscript = True
    run = p_eq2.add_run('t+1')
    run.font.superscript = True

    p = doc.add_paragraph()
    p.add_run('其中，ω为惯性权重，c')
    p.add_run('1').font.subscript = True
    p.add_run('、c')
    p.add_run('2').font.subscript = True
    p.add_run('为学习因子，r')
    p.add_run('1').font.subscript = True
    p.add_run('、r')
    p.add_run('2').font.subscript = True
    p.add_run('为[0,1]均匀随机数向量。为避免粒子过度飞出边界，通常对速度与位置进行截断：')
    p.add_run('v ∈ [-v')
    p.add_run('max').font.subscript = True
    p.add_run(', v')
    p.add_run('max').font.subscript = True
    p.add_run('], x ∈ [lower, upper]。')

    doc.add_heading('2.2 基本流程', level=2)
    doc.add_paragraph(
        '标准PSO流程可概括为：随机初始化粒子群；计算适应度并更新个体最优与全局最优；按更新公式迭代更新速度与位置，直至达到最大迭代次数或满足停止准则。')

    doc.add_heading('2.3 存在的问题', level=2)
    doc.add_paragraph(
        '当迭代进行到后期时，粒子位置趋于集中，多样性快速下降，群体容易围绕某个局部最优点震荡。若参数设置不当（如惯性权重过大或学习因子失衡），还可能出现收敛慢或搜索不稳定等现象。')

    # 3. 改进PSO方法
    doc.add_heading('3. 改进PSO方法', level=1)
    doc.add_paragraph(
        '本文在不改变PSO主体框架的前提下，引入三项改进：惯性权重递减、自适应学习因子、精英局部搜索。其核心目标是：前期保持较强探索能力，后期提升精细寻优与跳出停滞的能力。')

    doc.add_heading('3.1 线性递减惯性权重', level=2)
    doc.add_paragraph('采用线性递减策略：')

    # Formula 3
    p_eq3 = doc.add_paragraph()
    p_eq3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq3.add_run('ω(t) = ω')
    p_eq3.add_run('max').font.subscript = True
    p_eq3.add_run(' - (ω')
    p_eq3.add_run('max').font.subscript = True
    p_eq3.add_run(' - ω')
    p_eq3.add_run('min').font.subscript = True
    p_eq3.add_run(') · t / (T - 1)')

    doc.add_paragraph('其中T为最大迭代次数。ω在前期取较大值以增强全局搜索，后期减小以促进收敛。')

    doc.add_heading('3.2 成功率驱动的学习因子自适应', level=2)
    p = doc.add_paragraph('定义成功率 ')
    p.add_run('s').italic = True
    p.add_run(' 为本轮中个体最优得到改进的粒子比例。若 ')
    p.add_run('s').italic = True
    p.add_run(' 较低，说明搜索可能停滞，增大社会学习因子 c')
    p.add_run('2').font.subscript = True
    p.add_run(' 并适度减小 c')
    p.add_run('1').font.subscript = True
    p.add_run('，促使粒子向全局最优聚拢；若 ')
    p.add_run('s').italic = True
    p.add_run(' 较高，则适度增大 c')
    p.add_run('1').font.subscript = True
    p.add_run(' 以鼓励个体探索。')

    doc.add_heading('3.3 精英局部搜索', level=2)
    p = doc.add_paragraph('对当前全局最优解 ')
    p.add_run('g').italic = True
    p.add_run(' 引入概率递增的局部扰动：以概率 ')
    p.add_run('p').italic = True
    p.add_run('ls').font.subscript = True
    p.add_run('(t) 生成候选解 ')
    p.add_run('g\' = g + N(0, σ').italic = True
    p.add_run('2').font.superscript = True
    p.add_run('I)').italic = True
    p.add_run('，若 g\' 优于 g 则接受更新。步长 σ 与群体多样性相关：')

    p_eq4 = doc.add_paragraph()
    p_eq4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq4.add_run('σ = 0.05 · mean(std(x))')

    doc.add_heading('3.4 改进算法伪代码', level=2)
    doc.add_paragraph('输入：目标函数 f(x)，边界 [l,u]，粒子数 N，维度 D，迭代次数 T\n'
                      '初始化：随机生成 x, v；计算适应度；设置 p_i ← x_i，g ← argmin f(p_i)\n'
                      'for t = 1..T do\n'
                      '  计算 ω(t)；生成 r1, r2\n'
                      '  更新 v_i ← ω(t)v_i + c1 r1 (p_i - x_i) + c2 r2 (g - x_i)\n'
                      '  截断 v_i；更新 x_i ← x_i + v_i；截断 x_i\n'
                      '  计算适应度；更新 p_i 与 g\n'
                      '  计算成功率 s；根据 s 自适应调整 c1, c2\n'
                      '  以概率 p_ls(t) 对 g 执行局部搜索并更新\n'
                      'end for\n'
                      '输出：g 及 f(g)')

    # 4. 算法分析
    doc.add_heading('4. 算法分析', level=1)
    doc.add_paragraph(
        '（1）时间复杂度：在每次迭代中，需要对N个粒子计算一次适应度，若目标函数计算量为O(D)，则总体复杂度约为O(N·D·T)。改进算法增加的局部搜索每轮最多额外评估1次目标函数，其开销可近似看作O(D·T)，相对主体计算量较小。\n'
        '（2）稳定性与收敛讨论：线性递减惯性权重有助于从“全局探索”平滑过渡到“局部开发”；成功率自适应调整使学习因子在停滞时增强社会学习、在进展良好时增强个体探索；精英局部搜索为全局最优解提供小幅随机扰动，可一定程度上帮助跳出局部极值并缓解震荡。')

    # 5. 仿真设计
    doc.add_heading('5. 仿真设计', level=1)
    doc.add_paragraph(
        '实验采用Python（NumPy）实现。为保证统计意义，每个函数独立运行30次，记录最终最优值、均值与标准差，并绘制平均收敛曲线。')

    doc.add_heading('5.1 基准函数', level=2)
    # Sphere
    p = doc.add_paragraph('1）Sphere (单峰): ')
    p.add_run('f(x) = Σ x').italic = True
    p.add_run('j').font.subscript = True
    p.add_run('2').font.superscript = True

    # Rosenbrock
    p = doc.add_paragraph('2）Rosenbrock (窄谷): ')
    p.add_run('f(x) = Σ [100(x').italic = True
    p.add_run('j+1').font.subscript = True
    p.add_run(' - x').italic = True
    p.add_run('j').font.subscript = True
    p.add_run('2').font.superscript = True
    p.add_run(')').italic = True
    p.add_run('2').font.superscript = True
    p.add_run(' + (1 - x').italic = True
    p.add_run('j').font.subscript = True
    p.add_run(')').italic = True
    p.add_run('2').font.superscript = True
    p.add_run(']')

    # Rastrigin
    p = doc.add_paragraph('3）Rastrigin (多峰): ')
    p.add_run('f(x) = 10D + Σ [x').italic = True
    p.add_run('j').font.subscript = True
    p.add_run('2').font.superscript = True
    p.add_run(' - 10cos(2πx').italic = True
    p.add_run('j').font.subscript = True
    p.add_run(')]')

    # Griewank
    p = doc.add_paragraph('4）Griewank (多峰): ')
    p.add_run('f(x) = 1 + Σ x').italic = True
    p.add_run('j').font.subscript = True
    p.add_run('2').font.superscript = True
    p.add_run('/4000 - ∏ cos(x').italic = True
    p.add_run('j').font.subscript = True
    p.add_run('/√j)')

    doc.add_heading('5.2 参数设置', level=2)
    p = doc.add_paragraph('粒子数N=40，维度D=30，最大迭代次数T=200。标准PSO取ω=0.7，c1=c2=2.0。速度上限设置为 ')
    p.add_run('v').italic = True
    p.add_run('max').font.subscript = True
    p.add_run(' = 0.05·(u-l)。改进PSO取 ω')
    p.add_run('max').font.subscript = True
    p.add_run('=0.9，ω')
    p.add_run('min').font.subscript = True
    p.add_run('=0.4；局部搜索概率 p')
    p.add_run('ls').font.subscript = True
    p.add_run('(t)=0.05+0.25·t/(T-1)。')

    # 6. 仿真结果与讨论
    doc.add_heading('6. 仿真结果与讨论', level=1)

    # Table
    table = doc.add_table(rows=5, cols=4)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = '函数'
    hdr_cells[1].text = 'PSO（均值±标准差）'
    hdr_cells[2].text = '改进PSO（均值±标准差）'
    hdr_cells[3].text = '提升倍数'

    data = [
        ('Sphere', '0.0754 ± 0.0247', '0.0207 ± 0.0088', '3.65'),
        ('Rosenbrock', '29.309 ± 1.133', '28.159 ± 1.057', '1.04'),
        ('Rastrigin', '44.756 ± 10.157', '38.520 ± 12.241', '1.16'),
        ('Griewank', '1.259 ± 0.0849', '1.070 ± 0.0313', '1.18')
    ]

    for i, row_data in enumerate(data):
        row_cells = table.rows[i + 1].cells
        for j, cell_data in enumerate(row_data):
            row_cells[j].text = cell_data

    doc.add_paragraph(
        '表1 四个基准函数上的最终结果统计（30次独立运行，目标为最小化）').alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph('\n[此处应为收敛曲线图，见原图1]\n').alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph('图1 PSO与改进PSO在四个基准函数上的平均收敛曲线').alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        '从表1可见，改进PSO在四个测试函数上均取得更小的平均目标值，其中在Sphere函数上提升最为明显；在Rastrigin与Griewank等多峰函数上，改进算法的均值也有不同程度降低。从图1的收敛曲线看，改进PSO在后期通常能继续下降，体现了精英局部搜索与自适应参数在缓解停滞方面的作用。')

    # 7. 结论与展望
    doc.add_heading('7. 结论与展望', level=1)
    doc.add_paragraph(
        '本文针对标准PSO易早熟与后期停滞的问题，提出了结合线性递减惯性权重、成功率自适应学习因子与精英局部搜索的改进策略，并在四个经典基准函数上进行了仿真验证。结果表明，改进方法在相同计算预算下整体优于标准PSO。后续工作可进一步研究：更系统的参数敏感性分析、在约束优化/工程应用中的验证，以及与差分进化、遗传算法等方法的综合对比。')

    # 参考文献
    doc.add_heading('参考文献', level=1)
    doc.add_paragraph('[1] Kennedy J, Eberhart R. Particle swarm optimization. Proceedings of ICNN, 1995.')
    doc.add_paragraph('[2] Shi Y, Eberhart R. A modified particle swarm optimizer. Proceedings of ICEC, 1998.')
    doc.add_paragraph(
        '[3] Clerc M, Kennedy J. The particle swarm - explosion, stability, and convergence in a multidimensional complex space. IEEE Transactions on Evolutionary Computation, 2002.')

    file_path = 'Improved_PSO_Paper_Formatted.docx'
    doc.save(file_path)
    return file_path


file_path = create_formatted_doc()
