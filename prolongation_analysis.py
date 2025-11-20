import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

warnings.filterwarnings('ignore')

# Настройки
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def prepare_financial_data(financial_df):
    """Подготовка финансовых данных"""
    financial_df = financial_df.copy()
    month_columns = [col for col in financial_df.columns if
                     col not in ['id', 'Причина дубля', 'Account', 'Unnamed: 0']]

    def convert_to_float(value):
        if pd.isna(value) or value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            if value.lower() in ['стоп', 'stop', 'nan', '', 'в ноль', 'end']:
                return 0.0
            value_clean = re.sub(r'[^\d,.]', '', value.replace(' ', ''))
            value_clean = value_clean.replace(',', '.')
            try:
                return float(value_clean)
            except ValueError:
                return 0.0
        return 0.0

    for col in month_columns:
        financial_df[col] = financial_df[col].apply(convert_to_float)

    financial_long = pd.melt(
        financial_df,
        id_vars=['id', 'Причина дубля', 'Account'],
        value_vars=month_columns,
        var_name='month',
        value_name='shipment_amount'
    )

    def convert_russian_month(month_str):
        month_mapping = {
            'январь': '01', 'февраль': '02', 'март': '03', 'апрель': '04',
            'май': '05', 'июнь': '06', 'июль': '07', 'август': '08',
            'сентябрь': '09', 'октябрь': '10', 'ноябрь': '11', 'декабрь': '12'
        }
        try:
            parts = month_str.split()
            if len(parts) == 2:
                month_ru = parts[0].lower()
                year = parts[1]
                if month_ru in month_mapping:
                    month_num = month_mapping[month_ru]
                    return f"{year}-{month_num}"
        except Exception:
            pass
        return month_str

    financial_long['month'] = financial_long['month'].apply(convert_russian_month)
    financial_long = financial_long[financial_long['shipment_amount'] >= 0]
    financial_long = financial_long.sort_values('shipment_amount', ascending=False)
    financial_long = financial_long.drop_duplicates(['id', 'month'], keep='first')

    return financial_long


def get_previous_month(month):
    """Получение предыдущего месяца в формате YYYY-MM"""
    try:
        year = int(month.split('-')[0])
        month_num = int(month.split('-')[1])
        if month_num == 1:
            return f"{year - 1}-12"
        else:
            return f"{year}-{month_num - 1:02d}"
    except Exception:
        return month


def get_next_month(month):
    """Получение следующего месяца в формате YYYY-MM"""
    try:
        year = int(month.split('-')[0])
        month_num = int(month.split('-')[1])
        if month_num == 12:
            return f"{year + 1}-01"
        else:
            return f"{year}-{month_num + 1:02d}"
    except Exception:
        return month


def get_shipment_amount(project_id, month, financial_long_data):
    """Получение суммы отгрузки проекта в указанном месяце"""
    shipment = financial_long_data[
        (financial_long_data['id'] == project_id) &
        (financial_long_data['month'] == month)
        ]['shipment_amount']
    return shipment.sum() if not shipment.empty else 0.0


def get_projects_with_shipment_in_month(month, financial_long_data):
    """Получение проектов, имевших отгрузки в указанном месяце"""
    projects = financial_long_data[
        (financial_long_data['month'] == month) &
        (financial_long_data['shipment_amount'] > 0)
        ]['id'].unique()
    return list(projects)


def calculate_second_prolongation_coefficient_corrected(month, financial_long_data):
    """
    ПРАВИЛЬНЫЙ расчет коэффициента пролонгации во второй месяц
    Пример для мая: проекты с отгрузкой в марте, без отгрузки в апреле, но с отгрузкой в мае
    """
    # Месяцы для анализа
    completion_month = get_previous_month(get_previous_month(month))  # март для мая
    first_prolongation_month = get_previous_month(month)  # апрель для мая
    second_prolongation_month = month  # май для мая

    print(f"\n🔍 ПРАВИЛЬНЫЙ расчет второго коэффициента для {month}:")
    print(f"   Отгрузки были в: {completion_month}")
    print(f"   Пропустили месяц: {first_prolongation_month}")
    print(f"   Вернулись в: {second_prolongation_month}")

    # 1. Находим проекты, имевшие отгрузки в completion_month
    projects_with_completion_shipment = get_projects_with_shipment_in_month(completion_month, financial_long_data)
    print(f"   Проектов с отгрузками в {completion_month}: {len(projects_with_completion_shipment)}")

    # 2. Исключаем проекты, которые имели отгрузки в первый месяц пролонгации
    projects_without_first_prolongation = []
    for project in projects_with_completion_shipment:
        first_prolongation_amount = get_shipment_amount(project, first_prolongation_month, financial_long_data)
        if first_prolongation_amount == 0:  # Нет отгрузки в первый месяц пролонгации
            projects_without_first_prolongation.append(project)

    print(f"   Проектов БЕЗ отгрузки в {first_prolongation_month}: {len(projects_without_first_prolongation)}")

    # 3. Считаем сумму отгрузок в completion_month для этих проектов
    total_completion_amount = 0
    completion_shipments = []
    for project in projects_without_first_prolongation:
        completion_amount = get_shipment_amount(project, completion_month, financial_long_data)
        total_completion_amount += completion_amount
        completion_shipments.append((project, completion_amount))

    # 4. Считаем сумму отгрузок во второй месяц пролонгации
    total_second_prolongation_amount = 0
    prolonged_projects_second = []
    for project in projects_without_first_prolongation:
        second_prolongation_amount = get_shipment_amount(project, second_prolongation_month, financial_long_data)
        if second_prolongation_amount > 0:
            total_second_prolongation_amount += second_prolongation_amount
            prolonged_projects_second.append((project, second_prolongation_amount))

    print(f"   Пролонгировано во второй месяц: {len(prolonged_projects_second)}")
    print(f"   Сумма отгрузок в {completion_month}: {total_completion_amount:,.0f}")
    print(f"   Сумма пролонгации во второй месяц: {total_second_prolongation_amount:,.0f}")

    # Детальная информация для отладки
    if prolonged_projects_second:
        print(f"   📋 Примеры пролонгированных проектов: {[p[0] for p in prolonged_projects_second[:3]]}")

    # 5. Расчет коэффициента
    if total_completion_amount > 0:
        coefficient = (total_second_prolongation_amount / total_completion_amount) * 100
        print(f"   📊 Второй коэффициент пролонгации: {coefficient:.2f}%")
    else:
        coefficient = 0
        print(f"   📊 Второй коэффициент пролонгации: 0.00% (нет отгрузок в базовом месяце)")

    return {
        'month': month,
        'completion_month': completion_month,
        'first_prolongation_month': first_prolongation_month,
        'projects_count': len(projects_without_first_prolongation),
        'prolonged_count_second': len(prolonged_projects_second),
        'total_completion_amount': total_completion_amount,
        'total_second_prolongation_amount': total_second_prolongation_amount,
        'coefficient_second': coefficient,
        'prolonged_projects': [p[0] for p in prolonged_projects_second]
    }


def calculate_first_prolongation_coefficient(financial_long_data):
    """Расчет первого коэффициента пролонгации"""
    print("\n" + "=" * 60)
    print("🧮 РАСЧЕТ ПЕРВОГО КОЭФФИЦИЕНТА ПРОЛОНГАЦИИ")
    print("=" * 60)

    results_list = []
    all_months = sorted(financial_long_data['month'].unique())

    for i, current_month in enumerate(all_months[1:], 1):  # Начиная со второго месяца
        prev_month = all_months[i - 1]

        print(f"\n📅 Анализ месяца: {current_month}")
        print(f"   Проекты с отгрузками в: {prev_month}")

        # Проекты, которые имели отгрузки в предыдущем месяце
        projects_with_prev_shipment = get_projects_with_shipment_in_month(prev_month, financial_long_data)
        print(f"   Проектов с отгрузками в {prev_month}: {len(projects_with_prev_shipment)}")

        # Проекты, которые продолжились в текущем месяце
        continued_projects = []
        continued_shipment = 0
        for project in projects_with_prev_shipment:
            current_shipment = get_shipment_amount(project, current_month, financial_long_data)
            if current_shipment > 0:
                continued_projects.append(project)
                continued_shipment += current_shipment

        print(f"   Пролонгировано проектов: {len(continued_projects)}")

        # Суммы для расчета коэффициента
        total_prev_shipment = 0
        for project in projects_with_prev_shipment:
            total_prev_shipment += get_shipment_amount(project, prev_month, financial_long_data)

        print(f"   Сумма отгрузок в {prev_month}: {total_prev_shipment:,.0f}")
        print(f"   Сумма пролонгированных отгрузок: {continued_shipment:,.0f}")

        if total_prev_shipment > 0:
            prolongation_rate = continued_shipment / total_prev_shipment
            print(f"   📊 Коэффициент пролонгации: {prolongation_rate:.2%}")
        else:
            prolongation_rate = 0
            print(f"   📊 Коэффициент пролонгации: 0.00% (нет отгрузок в предыдущем месяце)")

        # Собираем результаты
        month_result = {
            'month': current_month,
            'previous_month': prev_month,
            'projects_with_prev_shipment': len(projects_with_prev_shipment),
            'prolongated_projects': len(continued_projects),
            'total_prev_shipment': total_prev_shipment,
            'prolongated_shipment': continued_shipment,
            'prolongation_rate': prolongation_rate
        }
        results_list.append(month_result)

    return pd.DataFrame(results_list)


def calculate_manager_prolongation_metrics(financial_long_data, prolongations_data):
    """
    Расчет коэффициентов пролонгации по каждому менеджеру
    """
    print("\n" + "=" * 60)
    print("👥 РАСЧЕТ КОЭФФИЦИЕНТОВ ПО МЕНЕДЖЕРАМ")
    print("=" * 60)

    # Объединяем данные для получения информации о менеджерах
    project_managers = prolongations_data[['id', 'AM']].drop_duplicates()

    # Добавляем информацию о менеджерах в финансовые данные
    financial_with_managers = financial_long_data.merge(
        project_managers,
        on='id',
        how='left',
        suffixes=('', '_prolongation')
    )

    # Заполняем пропущенные значения
    financial_with_managers['AM'] = financial_with_managers['AM'].fillna('без А/М')

    # Анализируем только 2023 год
    analysis_months_2023 = [month for month in sorted(financial_long_data['month'].unique())
                            if month.startswith('2023')]

    manager_results_list = []

    for month in analysis_months_2023[:6]:  # Анализируем первые 6 месяцев 2023
        prev_month = get_previous_month(month)

        print(f"\n📅 Анализ месяца {month}:")

        # Для каждого менеджера считаем коэффициенты
        managers = financial_with_managers['AM'].unique()

        for manager in managers:
            # Проекты менеджера, которые имели отгрузки в предыдущем месяце
            manager_projects_prev = financial_with_managers[
                (financial_with_managers['AM'] == manager) &
                (financial_with_managers['month'] == prev_month) &
                (financial_with_managers['shipment_amount'] > 0)
                ]['id'].unique()

            if len(manager_projects_prev) > 0:
                # Проекты, которые продолжились в текущем месяце
                continued_projects = financial_with_managers[
                    (financial_with_managers['AM'] == manager) &
                    (financial_with_managers['id'].isin(manager_projects_prev)) &
                    (financial_with_managers['month'] == month) &
                    (financial_with_managers['shipment_amount'] > 0)
                    ]

                # Суммы для расчета коэффициента
                total_prev_shipment = financial_with_managers[
                    (financial_with_managers['AM'] == manager) &
                    (financial_with_managers['id'].isin(manager_projects_prev)) &
                    (financial_with_managers['month'] == prev_month)
                    ]['shipment_amount'].sum()

                continued_shipment = continued_projects['shipment_amount'].sum()

                if total_prev_shipment > 0:
                    prolongation_rate = (continued_shipment / total_prev_shipment) * 100
                else:
                    prolongation_rate = 0

                manager_results_list.append({
                    'month': month,
                    'manager': manager,
                    'projects_with_prev_shipment': len(manager_projects_prev),
                    'prolongated_projects': len(continued_projects),
                    'total_prev_shipment': total_prev_shipment,
                    'prolongated_shipment': continued_shipment,
                    'prolongation_rate': prolongation_rate
                })

                if prolongation_rate > 0:
                    print(
                        f"   👤 {manager}: {prolongation_rate:.1f}% ({len(continued_projects)}/{len(manager_projects_prev)} проектов)")

    return pd.DataFrame(manager_results_list)


def create_visualizations(first_coeff_results, second_coeff_results):
    """Создание визуализаций"""
    print("\n" + "=" * 60)
    print("📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    if len(first_coeff_results) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # График 1: Динамика первого коэффициента пролонгации
        ax1.plot(first_coeff_results['month'], first_coeff_results['prolongation_rate'],
                 marker='o', linewidth=2, markersize=6, color='blue')
        ax1.set_title('Динамика первого коэффициента пролонгации', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Месяц')
        ax1.set_ylabel('Коэффициент пролонгации')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Добавляем значения на график
        for i, row in first_coeff_results.iterrows():
            ax1.annotate(f'{row["prolongation_rate"]:.1%}',
                         (row['month'], row['prolongation_rate']),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        # График 2: Количество пролонгированных проектов
        ax2.bar(first_coeff_results['month'], first_coeff_results['prolongated_projects'],
                alpha=0.7, color='green')
        ax2.set_title('Количество пролонгированных проектов (1-й коэффициент)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Месяц')
        ax2.set_ylabel('Количество проектов')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # Добавляем значения на столбцы
        for i, v in enumerate(first_coeff_results['prolongated_projects']):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)

        # График 3: Второй коэффициент пролонгации
        if second_coeff_results:
            second_coeff_df = pd.DataFrame(second_coeff_results)
            ax3.bar(second_coeff_df['month'], second_coeff_df['coefficient_second'] / 100,
                    alpha=0.7, color='orange')
            ax3.set_title('Второй коэффициент пролонгации', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Месяц')
            ax3.set_ylabel('Коэффициент пролонгации')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)

            # Добавляем значения на столбцы
            for i, row in second_coeff_df.iterrows():
                ax3.text(i, row['coefficient_second'] / 100 + 0.02,
                         f'{row["coefficient_second"]:.1f}%',
                         ha='center', va='bottom', fontsize=9)

        # График 4: Сравнение объемов отгрузок
        ax4.bar(first_coeff_results['month'], first_coeff_results['total_prev_shipment'] / 1000000,
                alpha=0.6, label='Общие отгрузки', color='blue')
        ax4.bar(first_coeff_results['month'], first_coeff_results['prolongated_shipment'] / 1000000,
                alpha=0.8, label='Пролонгированные', color='red')
        ax4.set_title('Объемы отгрузок (млн руб.)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Месяц')
        ax4.set_ylabel('Сумма отгрузок, млн руб.')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('improved_prolongation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Графики сохранены в improved_prolongation_analysis.png")


def create_comprehensive_report(first_coeff_results, second_coeff_results, manager_results, financial_long_data):
    """
    Создание комплексного отчета
    """
    print("\n" + "=" * 60)
    print("💾 СОЗДАНИЕ КОМПЛЕКСНОГО ОТЧЕТА")
    print("=" * 60)

    with pd.ExcelWriter('comprehensive_prolongation_report.xlsx', engine='openpyxl') as writer:

        # 1. Сводка по отделу
        summary_data = {
            'Показатель': [
                'Средний коэффициент пролонгации (1-й)',
                'Средний коэффициент пролонгации (2-й)',
                'Всего пролонгировано проектов',
                'Общий объем пролонгированных отгрузок',
                'Период анализа',
                'Количество менеджеров'
            ],
            'Значение': [
                f"{first_coeff_results['prolongation_rate'].mean() * 100:.2f}%" if len(
                    first_coeff_results) > 0 else "0.00%",
                f"{pd.DataFrame(second_coeff_results)['coefficient_second'].mean():.2f}%" if second_coeff_results else "0.00%",
                f"{first_coeff_results['prolongated_projects'].sum()}" if len(first_coeff_results) > 0 else "0",
                f"{first_coeff_results['prolongated_shipment'].sum():,.0f} руб." if len(
                    first_coeff_results) > 0 else "0 руб.",
                f"{first_coeff_results['month'].min()} - {first_coeff_results['month'].max()}" if len(
                    first_coeff_results) > 0 else "Нет данных",
                f"{manager_results['manager'].nunique()}" if len(manager_results) > 0 else "0"
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Сводка по отделу', index=False)

        # 2. Детальные результаты по месяцам (1-й коэффициент)
        if len(first_coeff_results) > 0:
            results_with_percent = first_coeff_results.copy()
            results_with_percent['prolongation_rate_percent'] = results_with_percent['prolongation_rate'] * 100
            results_with_percent[['month', 'previous_month', 'projects_with_prev_shipment',
                                  'prolongated_projects', 'total_prev_shipment', 'prolongated_shipment',
                                  'prolongation_rate_percent']].to_excel(writer, sheet_name='1-й коэффициент',
                                                                         index=False)

        # 3. Второй коэффициент пролонгации
        if second_coeff_results:
            second_coeff_df = pd.DataFrame(second_coeff_results)
            second_coeff_df.to_excel(writer, sheet_name='2-й коэффициент', index=False)

        # 4. Результаты по менеджерам
        if len(manager_results) > 0:
            manager_summary = manager_results.groupby('manager').agg({
                'prolongation_rate': 'mean',
                'projects_with_prev_shipment': 'sum',
                'prolongated_projects': 'sum',
                'total_prev_shipment': 'sum',
                'prolongated_shipment': 'sum'
            }).reset_index()
            manager_summary['prolongation_rate'] = manager_summary['prolongation_rate'].round(2)
            manager_summary.to_excel(writer, sheet_name='Итоги по менеджерам', index=False)

            # Детальные данные по менеджерам
            manager_details = manager_results.copy()
            manager_details['prolongation_rate'] = manager_details['prolongation_rate'].round(2)
            manager_details.to_excel(writer, sheet_name='Детали по менеджерам', index=False)

        # 5. Топ менеджеров
        if len(manager_results) > 0:
            top_managers = manager_results.groupby('manager')['prolongation_rate'].mean().nlargest(5)
            pd.DataFrame({
                'Менеджер': top_managers.index,
                'Средний коэффициент': top_managers.values.round(2)
            }).to_excel(writer, sheet_name='Топ менеджеров', index=False)

        # 6. Исходные данные
        financial_long_data.head(1000).to_excel(writer, sheet_name='Исходные данные', index=False)

    print("✅ Комплексный отчет сохранен в comprehensive_prolongation_report.xlsx")


def calculate_complete_prolongation_analysis():
    """Полный анализ пролонгации с исправленной логикой"""
    print("🚀 ЗАПУСК ИСПРАВЛЕННОГО АНАЛИЗА ПРОЛОНГАЦИЙ")
    print("=" * 60)

    # Загрузка данных
    prolongations_data = pd.read_csv('prolongations.csv')
    financial_data = pd.read_csv('financial_data.csv')

    # Подготовка данных
    financial_long_prepared = prepare_financial_data(financial_data)

    # Расчет первого коэффициента пролонгации
    first_coeff_results = calculate_first_prolongation_coefficient(financial_long_prepared)

    # Расчет второго коэффициента пролонгации (ИСПРАВЛЕННЫЙ)
    print("\n" + "=" * 60)
    print("🔄 РАСЧЕТ ВТОРОГО КОЭФФИЦИЕНТА ПРОЛОНГАЦИИ (ИСПРАВЛЕННЫЙ)")
    print("=" * 60)

    second_coeff_results_list = []
    analysis_months_2023 = [month for month in sorted(financial_long_prepared['month'].unique())
                            if month.startswith('2023')]

    for month in analysis_months_2023[:6]:  # Анализируем первые 6 месяцев 2023
        try:
            second_coeff_data = calculate_second_prolongation_coefficient_corrected(month, financial_long_prepared)
            second_coeff_results_list.append(second_coeff_data)
        except Exception as e:
            print(f"❌ Ошибка при расчете второго коэффициента для {month}: {e}")

    # Расчет по менеджерам
    manager_results_df = calculate_manager_prolongation_metrics(financial_long_prepared, prolongations_data)

    # Визуализация результатов
    create_visualizations(first_coeff_results, second_coeff_results_list)

    # Создание комплексного отчета
    create_comprehensive_report(first_coeff_results, second_coeff_results_list, manager_results_df,
                                financial_long_prepared)

    # Сводная статистика
    print("\n" + "=" * 60)
    print("📈 СВОДНАЯ СТАТИСТИКА")
    print("=" * 60)

    if len(first_coeff_results) > 0:
        avg_prolongation_rate = first_coeff_results['prolongation_rate'].mean()
        total_prolongated_projects = first_coeff_results['prolongated_projects'].sum()
        total_prolongated_shipment = first_coeff_results['prolongated_shipment'].sum()

        print(f"📊 ОБЩИЕ РЕЗУЛЬТАТЫ:")
        print(f"   • Средний коэффициент пролонгации (1-й): {avg_prolongation_rate:.2%}")
        print(f"   • Всего пролонгировано проектов: {total_prolongated_projects}")
        print(f"   • Общий объем пролонгированных отгрузок: {total_prolongated_shipment:,.0f} руб.")
        print(
            f"   • Анализированный период: {first_coeff_results['month'].min()} - {first_coeff_results['month'].max()}")

        if second_coeff_results_list:
            second_avg = pd.DataFrame(second_coeff_results_list)['coefficient_second'].mean()
            print(f"   • Средний коэффициент пролонгации (2-й): {second_avg:.2f}%")

        # Лучшие месяцы по пролонгации
        best_months = first_coeff_results.nlargest(3, 'prolongation_rate')
        print(f"\n🏆 ЛУЧШИЕ МЕСЯЦЫ ПО ПРОЛОНГАЦИИ:")
        for _, row in best_months.iterrows():
            print(f"   • {row['month']}: {row['prolongation_rate']:.2%} ({row['prolongated_projects']} проектов)")

    return first_coeff_results, second_coeff_results_list, manager_results_df


# ЗАПУСК ПРОГРАММЫ
if __name__ == "__main__":
    first_coeff, second_coeff, manager_results = calculate_complete_prolongation_analysis()

    print("\n🎉 ПОЛНЫЙ АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 60)
    print("Созданные файлы:")
    print("  1. improved_prolongation_analysis.png - Графики анализа")
    print("  2. comprehensive_prolongation_report.xlsx - Полный отчет")
    print("\n📊 ОСНОВНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"  • Проанализировано месяцев: {len(first_coeff)}")
    print(f"  • Рассчитано вторых коэффициентов: {len(second_coeff)}")
    print(f"  • Проанализировано менеджеров: {manager_results['manager'].nunique() if len(manager_results) > 0 else 0}")

    if second_coeff:
        print(f"  • Второй коэффициент показывает проекты, которые 'вернулись' после пропуска месяца")

    print("\n📈 РЕКОМЕНДАЦИИ ДЛЯ РУКОВОДИТЕЛЯ:")
    print("  • Используйте comprehensive_prolongation_report.xlsx для детального анализа")
    print("  • Сравните эффективность менеджеров по коэффициентам пролонгации")
    print("  • Проанализируйте причины различий в первом и втором коэффициентах")
    print("  • Обратите внимание на проекты, которые 'возвращаются' после перерыва")
    print("  • Разработайте план улучшения на основе выявленных закономерностей")
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# import re
#
# warnings.filterwarnings('ignore')
#
# # Настройки
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# plt.style.use('seaborn-v0_8')
# sns.set_palette("husl")
#
#
# def prepare_financial_data(financial_df):
#     """Подготовка финансовых данных"""
#     financial_df = financial_df.copy()
#     month_columns = [col for col in financial_df.columns if
#                      col not in ['id', 'Причина дубля', 'Account', 'Unnamed: 0']]
#
#     def convert_to_float(value):
#         if pd.isna(value) or value is None:
#             return 0.0
#         if isinstance(value, (int, float)):
#             return float(value)
#         if isinstance(value, str):
#             if value.lower() in ['стоп', 'stop', 'nan', '', 'в ноль', 'end']:
#                 return 0.0
#             value_clean = re.sub(r'[^\d,.]', '', value.replace(' ', ''))
#             value_clean = value_clean.replace(',', '.')
#             try:
#                 return float(value_clean)
#             except ValueError:
#                 return 0.0
#         return 0.0
#
#     for col in month_columns:
#         financial_df[col] = financial_df[col].apply(convert_to_float)
#
#     financial_long = pd.melt(
#         financial_df,
#         id_vars=['id', 'Причина дубля', 'Account'],
#         value_vars=month_columns,
#         var_name='month',
#         value_name='shipment_amount'
#     )
#
#     def convert_russian_month(month_str):
#         month_mapping = {
#             'январь': '01', 'февраль': '02', 'март': '03', 'апрель': '04',
#             'май': '05', 'июнь': '06', 'июль': '07', 'август': '08',
#             'сентябрь': '09', 'октябрь': '10', 'ноябрь': '11', 'декабрь': '12'
#         }
#         try:
#             parts = month_str.split()
#             if len(parts) == 2:
#                 month_ru = parts[0].lower()
#                 year = parts[1]
#                 if month_ru in month_mapping:
#                     month_num = month_mapping[month_ru]
#                     return f"{year}-{month_num}"
#         except Exception:
#             pass
#         return month_str
#
#     financial_long['month'] = financial_long['month'].apply(convert_russian_month)
#     financial_long = financial_long[financial_long['shipment_amount'] >= 0]
#     financial_long = financial_long.sort_values('shipment_amount', ascending=False)
#     financial_long = financial_long.drop_duplicates(['id', 'month'], keep='first')
#
#     return financial_long
#
#
# def get_previous_month(month):
#     """Получение предыдущего месяца в формате YYYY-MM"""
#     try:
#         year = int(month.split('-')[0])
#         month_num = int(month.split('-')[1])
#         if month_num == 1:
#             return f"{year - 1}-12"
#         else:
#             return f"{year}-{month_num - 1:02d}"
#     except Exception:
#         return month
#
#
# def get_completed_projects(completion_month, financial_long_data):
#     """Получение проектов, завершившихся в указанном месяце"""
#     # Находим проекты, у которых completion_month - последний месяц с отгрузкой
#     project_last_months = financial_long_data[financial_long_data['shipment_amount'] > 0].groupby('id')['month'].max()
#     completed_projects = project_last_months[project_last_months == completion_month].index.tolist()
#     return completed_projects
#
#
# def get_shipment_amount(project_id, month, financial_long_data):
#     """Получение суммы отгрузки проекта в указанном месяце"""
#     shipment = financial_long_data[
#         (financial_long_data['id'] == project_id) &
#         (financial_long_data['month'] == month)
#         ]['shipment_amount']
#     return shipment.sum() if not shipment.empty else 0.0
#
#
# def calculate_second_prolongation_coefficient(month, financial_long_data):
#     """
#     Расчет коэффициента пролонгации во второй месяц
#     Пример для мая: проекты завершившиеся в марте, без отгрузки в апреле, но с отгрузкой в мае
#     """
#     # Месяцы для анализа
#     completion_month = get_previous_month(get_previous_month(month))  # март для мая
#     first_prolongation_month = get_previous_month(month)  # апрель для мая
#     second_prolongation_month = month  # май для мая
#
#     print(f"\n🔍 Расчет второго коэффициента для {month}:")
#     print(f"   Завершились в: {completion_month}")
#     print(f"   Первая пролонгация: {first_prolongation_month}")
#     print(f"   Вторая пролонгация: {second_prolongation_month}")
#
#     # 1. Находим проекты, завершившиеся в completion_month
#     completed_projects = get_completed_projects(completion_month, financial_long_data)
#     print(f"   Проектов завершилось в {completion_month}: {len(completed_projects)}")
#
#     # 2. Исключаем проекты, которые уже пролонгированы в первый месяц
#     projects_without_first_prolongation = []
#     for project in completed_projects:
#         first_prolongation_amount = get_shipment_amount(project, first_prolongation_month, financial_long_data)
#         if first_prolongation_amount == 0:  # Нет пролонгации в первый месяц
#             projects_without_first_prolongation.append(project)
#
#     print(f"   Проектов без пролонгации в первый месяц: {len(projects_without_first_prolongation)}")
#
#     # 3. Считаем сумму отгрузок в completion_month для этих проектов
#     total_completion_amount = 0
#     for project in projects_without_first_prolongation:
#         completion_amount = get_shipment_amount(project, completion_month, financial_long_data)
#         total_completion_amount += completion_amount
#
#     # 4. Считаем сумму пролонгированных отгрузок во второй месяц
#     total_second_prolongation_amount = 0
#     prolonged_projects_second = []
#     for project in projects_without_first_prolongation:
#         second_prolongation_amount = get_shipment_amount(project, second_prolongation_month, financial_long_data)
#         if second_prolongation_amount > 0:
#             total_second_prolongation_amount += second_prolongation_amount
#             prolonged_projects_second.append(project)
#
#     print(f"   Пролонгировано во второй месяц: {len(prolonged_projects_second)}")
#     print(f"   Сумма отгрузок в {completion_month}: {total_completion_amount:,.0f}")
#     print(f"   Сумма пролонгации во второй месяц: {total_second_prolongation_amount:,.0f}")
#
#     # 5. Расчет коэффициента
#     if total_completion_amount > 0:
#         coefficient = (total_second_prolongation_amount / total_completion_amount) * 100
#         print(f"   📊 Второй коэффициент пролонгации: {coefficient:.2f}%")
#     else:
#         coefficient = 0
#         print(f"   📊 Второй коэффициент пролонгации: 0.00%")
#
#     return {
#         'month': month,
#         'completion_month': completion_month,
#         'first_prolongation_month': first_prolongation_month,
#         'projects_count': len(projects_without_first_prolongation),
#         'prolonged_count_second': len(prolonged_projects_second),
#         'total_completion_amount': total_completion_amount,
#         'total_second_prolongation_amount': total_second_prolongation_amount,
#         'coefficient_second': coefficient
#     }
#
#
# def calculate_manager_prolongation_metrics(financial_long_data, prolongations_data):
#     """
#     Расчет коэффициентов пролонгации по каждому менеджеру
#     """
#     print("\n" + "=" * 60)
#     print("👥 РАСЧЕТ КОЭФФИЦИЕНТОВ ПО МЕНЕДЖЕРАМ")
#     print("=" * 60)
#
#     # Объединяем данные для получения информации о менеджерах
#     project_managers = prolongations_data[['id', 'AM']].drop_duplicates()
#
#     # Добавляем информацию о менеджерах в финансовые данные
#     financial_with_managers = financial_long_data.merge(
#         project_managers,
#         on='id',
#         how='left',
#         suffixes=('', '_prolongation')
#     )
#
#     # Заполняем пропущенные значения
#     financial_with_managers['AM'] = financial_with_managers['AM'].fillna('без А/М')
#
#     # Анализируем только 2023 год
#     analysis_months_2023 = [month for month in sorted(financial_long_data['month'].unique())
#                             if month.startswith('2023')]
#
#     manager_results_list = []
#
#     for month in analysis_months_2023[:6]:  # Анализируем первые 6 месяцев 2023 для примера
#         prev_month = get_previous_month(month)
#
#         print(f"\n📅 Анализ месяца {month}:")
#
#         # Для каждого менеджера считаем коэффициенты
#         managers = financial_with_managers['AM'].unique()
#
#         for manager in managers:
#             # Проекты менеджера, которые имели отгрузки в предыдущем месяце
#             manager_projects_prev = financial_with_managers[
#                 (financial_with_managers['AM'] == manager) &
#                 (financial_with_managers['month'] == prev_month) &
#                 (financial_with_managers['shipment_amount'] > 0)
#                 ]['id'].unique()
#
#             if len(manager_projects_prev) > 0:
#                 # Проекты, которые продолжились в текущем месяце
#                 continued_projects = financial_with_managers[
#                     (financial_with_managers['AM'] == manager) &
#                     (financial_with_managers['id'].isin(manager_projects_prev)) &
#                     (financial_with_managers['month'] == month) &
#                     (financial_with_managers['shipment_amount'] > 0)
#                     ]
#
#                 # Суммы для расчета коэффициента
#                 total_prev_shipment = financial_with_managers[
#                     (financial_with_managers['AM'] == manager) &
#                     (financial_with_managers['id'].isin(manager_projects_prev)) &
#                     (financial_with_managers['month'] == prev_month)
#                     ]['shipment_amount'].sum()
#
#                 continued_shipment = continued_projects['shipment_amount'].sum()
#
#                 if total_prev_shipment > 0:
#                     prolongation_rate = (continued_shipment / total_prev_shipment) * 100
#                 else:
#                     prolongation_rate = 0
#
#                 manager_results_list.append({
#                     'month': month,
#                     'manager': manager,
#                     'projects_with_prev_shipment': len(manager_projects_prev),
#                     'prolongated_projects': len(continued_projects),
#                     'total_prev_shipment': total_prev_shipment,
#                     'prolongated_shipment': continued_shipment,
#                     'prolongation_rate': prolongation_rate
#                 })
#
#                 if prolongation_rate > 0:
#                     print(
#                         f"   👤 {manager}: {prolongation_rate:.1f}% ({len(continued_projects)}/{len(manager_projects_prev)} проектов)")
#
#     return pd.DataFrame(manager_results_list)
#
#
# def create_comprehensive_report(results_data, second_coeff_results_data, manager_results_data, financial_long_data):
#     """
#     Создание комплексного отчета
#     """
#     print("\n" + "=" * 60)
#     print("💾 СОЗДАНИЕ КОМПЛЕКСНОГО ОТЧЕТА")
#     print("=" * 60)
#
#     with pd.ExcelWriter('comprehensive_prolongation_report.xlsx', engine='openpyxl') as writer:
#
#         # 1. Сводка по отделу
#         summary_data = {
#             'Показатель': [
#                 'Средний коэффициент пролонгации (1-й)',
#                 'Средний коэффициент пролонгации (2-й)',
#                 'Всего пролонгировано проектов',
#                 'Общий объем пролонгированных отгрузок',
#                 'Период анализа',
#                 'Количество менеджеров'
#             ],
#             'Значение': [
#                 f"{results_data['prolongation_rate'].mean() * 100:.2f}%" if len(results_data) > 0 else "0.00%",
#                 f"{pd.DataFrame(second_coeff_results_data)['coefficient_second'].mean():.2f}%" if second_coeff_results_data else "0.00%",
#                 f"{results_data['prolongated_projects'].sum()}" if len(results_data) > 0 else "0",
#                 f"{results_data['prolongated_shipment'].sum():,.0f} руб." if len(results_data) > 0 else "0 руб.",
#                 f"{results_data['month'].min()} - {results_data['month'].max()}" if len(
#                     results_data) > 0 else "Нет данных",
#                 f"{manager_results_data['manager'].nunique()}" if len(manager_results_data) > 0 else "0"
#             ]
#         }
#         pd.DataFrame(summary_data).to_excel(writer, sheet_name='Сводка по отделу', index=False)
#
#         # 2. Детальные результаты по месяцам (1-й коэффициент)
#         if len(results_data) > 0:
#             results_with_percent = results_data.copy()
#             results_with_percent['prolongation_rate_percent'] = results_with_percent['prolongation_rate'] * 100
#             results_with_percent[['month', 'previous_month', 'projects_with_prev_shipment',
#                                   'prolongated_projects', 'total_prev_shipment', 'prolongated_shipment',
#                                   'prolongation_rate_percent']].to_excel(writer, sheet_name='1-й коэффициент',
#                                                                          index=False)
#
#         # 3. Второй коэффициент пролонгации
#         if second_coeff_results_data:
#             second_coeff_df = pd.DataFrame(second_coeff_results_data)
#             second_coeff_df.to_excel(writer, sheet_name='2-й коэффициент', index=False)
#
#         # 4. Результаты по менеджерам
#         if len(manager_results_data) > 0:
#             manager_summary = manager_results_data.groupby('manager').agg({
#                 'prolongation_rate': 'mean',
#                 'projects_with_prev_shipment': 'sum',
#                 'prolongated_projects': 'sum',
#                 'total_prev_shipment': 'sum',
#                 'prolongated_shipment': 'sum'
#             }).reset_index()
#             manager_summary['prolongation_rate'] = manager_summary['prolongation_rate'].round(2)
#             manager_summary.to_excel(writer, sheet_name='Итоги по менеджерам', index=False)
#
#             # Детальные данные по менеджерам
#             manager_details = manager_results_data.copy()
#             manager_details['prolongation_rate'] = manager_details['prolongation_rate'].round(2)
#             manager_details.to_excel(writer, sheet_name='Детали по менеджерам', index=False)
#
#         # 5. Топ менеджеров
#         if len(manager_results_data) > 0:
#             top_managers = manager_results_data.groupby('manager')['prolongation_rate'].mean().nlargest(5)
#             pd.DataFrame({
#                 'Менеджер': top_managers.index,
#                 'Средний коэффициент': top_managers.values.round(2)
#             }).to_excel(writer, sheet_name='Топ менеджеров', index=False)
#
#     print("✅ Комплексный отчет сохранен в comprehensive_prolongation_report.xlsx")
#
#
# def calculate_prolongation_metrics_improved():
#     """Расчет метрик пролонгации"""
#     print("🚀 ЗАПУСК УЛУЧШЕННОГО АНАЛИЗА ПРОЛОНГАЦИЙ")
#
#     # Загрузка данных
#     prolongations_data = pd.read_csv('prolongations.csv')
#     financial_data = pd.read_csv('financial_data.csv')
#
#     # Подготовка данных
#     financial_long_prepared = prepare_financial_data(financial_data)
#
#     # УЛУЧШЕННЫЙ РАСЧЕТ ПРОЛОНГАЦИЙ
#     print("\n" + "=" * 60)
#     print("🧮 УЛУЧШЕННЫЙ РАСЧЕТ КОЭФФИЦИЕНТОВ ПРОЛОНГАЦИИ")
#     print("=" * 60)
#
#     # Создаем расширенную временную линию
#     project_shipments = financial_long_prepared[financial_long_prepared['shipment_amount'] > 0].groupby('id').agg({
#         'month': list,
#         'shipment_amount': list,
#         'Account': 'first'
#     }).reset_index()
#
#     project_shipments['months'] = project_shipments['month'].apply(sorted)
#     project_shipments['last_shipment_month'] = project_shipments['months'].apply(lambda x: x[-1] if x else None)
#
#     results_list = []
#
#     # Анализируем каждый месяц
#     all_months = sorted(financial_long_prepared['month'].unique())
#
#     for i, current_month in enumerate(all_months[1:], 1):  # Начиная со второго месяца
#         prev_month = all_months[i - 1]
#
#         print(f"\n📅 Анализ месяца: {current_month}")
#         print(f"   Проекты завершились в: {prev_month}")
#
#         # Проекты, которые имели отгрузки в предыдущем месяце
#         projects_with_prev_shipment = financial_long_prepared[
#             (financial_long_prepared['month'] == prev_month) &
#             (financial_long_prepared['shipment_amount'] > 0)
#             ]['id'].unique()
#
#         print(f"   Проектов с отгрузками в {prev_month}: {len(projects_with_prev_shipment)}")
#
#         # Проекты, которые продолжились в текущем месяце
#         continued_projects = financial_long_prepared[
#             (financial_long_prepared['id'].isin(projects_with_prev_shipment)) &
#             (financial_long_prepared['month'] == current_month) &
#             (financial_long_prepared['shipment_amount'] > 0)
#             ]
#
#         print(f"   Пролонгировано проектов: {len(continued_projects)}")
#
#         # Суммы для расчета коэффициента
#         total_prev_shipment = financial_long_prepared[
#             (financial_long_prepared['id'].isin(projects_with_prev_shipment)) &
#             (financial_long_prepared['month'] == prev_month)
#             ]['shipment_amount'].sum()
#
#         continued_shipment = continued_projects['shipment_amount'].sum()
#
#         print(f"   Сумма отгрузок в {prev_month}: {total_prev_shipment:,.0f}")
#         print(f"   Сумма пролонгированных отгрузок: {continued_shipment:,.0f}")
#
#         if total_prev_shipment > 0:
#             prolongation_rate = continued_shipment / total_prev_shipment
#             print(f"   📊 Коэффициент пролонгации: {prolongation_rate:.2%}")
#         else:
#             prolongation_rate = 0
#             print(f"   📊 Коэффициент пролонгации: 0.00% (нет отгрузок в предыдущем месяце)")
#
#         # Собираем результаты
#         month_result = {
#             'month': current_month,
#             'previous_month': prev_month,
#             'projects_with_prev_shipment': len(projects_with_prev_shipment),
#             'prolongated_projects': len(continued_projects),
#             'total_prev_shipment': total_prev_shipment,
#             'prolongated_shipment': continued_shipment,
#             'prolongation_rate': prolongation_rate
#         }
#         results_list.append(month_result)
#
#     # Создаем DataFrame с результатами
#     results_df = pd.DataFrame(results_list)
#
#     # ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
#     print("\n" + "=" * 60)
#     print("📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
#     print("=" * 60)
#
#     if len(results_df) > 0:
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
#
#         # График 1: Динамика коэффициента пролонгации
#         ax1.plot(results_df['month'], results_df['prolongation_rate'], marker='o', linewidth=2, markersize=8)
#         ax1.set_title('Динамика коэффициента пролонгации по месяцам', fontsize=14, fontweight='bold')
#         ax1.set_xlabel('Месяц')
#         ax1.set_ylabel('Коэффициент пролонгации')
#         ax1.tick_params(axis='x', rotation=45)
#         ax1.grid(True, alpha=0.3)
#         ax1.set_ylim(0, 1)
#
#         # Добавляем значения на график
#         for i, row in results_df.iterrows():
#             ax1.annotate(f'{row["prolongation_rate"]:.1%}',
#                          (row['month'], row['prolongation_rate']),
#                          textcoords="offset points", xytext=(0, 10), ha='center')
#
#         # График 2: Количество пролонгированных проектов
#         ax2.bar(results_df['month'], results_df['prolongated_projects'], alpha=0.7)
#         ax2.set_title('Количество пролонгированных проектов по месяцам', fontsize=14, fontweight='bold')
#         ax2.set_xlabel('Месяц')
#         ax2.set_ylabel('Количество проектов')
#         ax2.tick_params(axis='x', rotation=45)
#         ax2.grid(True, alpha=0.3)
#
#         # Добавляем значения на столбцы
#         for i, v in enumerate(results_df['prolongated_projects']):
#             ax2.text(i, v + 0.1, str(v), ha='center', va='bottom')
#
#         plt.tight_layout()
#         plt.savefig('improved_prolongation_analysis.png', dpi=300, bbox_inches='tight')
#         plt.show()
#
#         print("✅ Графики сохранены в improved_prolongation_analysis.png")
#
#     # СВОДНАЯ СТАТИСТИКА
#     print("\n" + "=" * 60)
#     print("📈 СВОДНАЯ СТАТИСТИКА")
#     print("=" * 60)
#
#     avg_prolongation_rate = 0
#     total_prolongated_projects = 0
#     total_prolongated_shipment = 0
#
#     if len(results_df) > 0:
#         avg_prolongation_rate = results_df['prolongation_rate'].mean()
#         total_prolongated_projects = results_df['prolongated_projects'].sum()
#         total_prolongated_shipment = results_df['prolongated_shipment'].sum()
#
#         print(f"📊 ОБЩИЕ РЕЗУЛЬТАТЫ:")
#         print(f"   • Средний коэффициент пролонгации: {avg_prolongation_rate:.2%}")
#         print(f"   • Всего пролонгировано проектов: {total_prolongated_projects}")
#         print(f"   • Общий объем пролонгированных отгрузок: {total_prolongated_shipment:,.0f} руб.")
#         print(f"   • Анализированный период: {results_df['month'].min()} - {results_df['month'].max()}")
#
#         # Лучшие месяцы по пролонгации
#         best_months = results_df.nlargest(3, 'prolongation_rate')
#         print(f"\n🏆 ЛУЧШИЕ МЕСЯЦЫ ПО ПРОЛОНГАЦИИ:")
#         for _, row in best_months.iterrows():
#             print(f"   • {row['month']}: {row['prolongation_rate']:.2%} ({row['prolongated_projects']} проектов)")
#
#     # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
#     print("\n" + "=" * 60)
#     print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
#     print("=" * 60)
#
#     # Создаем отчет
#     report_data = {
#         'Показатель': [
#             'Средний коэффициент пролонгации',
#             'Всего пролонгировано проектов',
#             'Общий объем пролонгированных отгрузок',
#             'Период анализа',
#             'Количество анализируемых месяцев'
#         ],
#         'Значение': [
#             f"{avg_prolongation_rate:.2%}" if len(results_df) > 0 else "0.00%",
#             f"{total_prolongated_projects}" if len(results_df) > 0 else "0",
#             f"{total_prolongated_shipment:,.0f} руб." if len(results_df) > 0 else "0 руб.",
#             f"{results_df['month'].min()} - {results_df['month'].max()}" if len(results_df) > 0 else "Нет данных",
#             f"{len(results_df)}" if len(results_df) > 0 else "0"
#         ]
#     }
#
#     report_df = pd.DataFrame(report_data)
#
#     with pd.ExcelWriter('improved_prolongation_report.xlsx', engine='openpyxl') as writer:
#         report_df.to_excel(writer, sheet_name='Сводка', index=False)
#         if len(results_df) > 0:
#             results_df.to_excel(writer, sheet_name='Детальные результаты', index=False)
#
#         # Добавляем исходные данные
#         financial_long_prepared.head(1000).to_excel(writer, sheet_name='Исходные данные', index=False)
#
#     print("✅ Отчет сохранен в improved_prolongation_report.xlsx")
#
#     return results_df
#
#
# # ДОПОЛНЯЕМ ОСНОВНУЮ ФУНКЦИЮ
# def calculate_complete_prolongation_analysis():
#     """Полный анализ пролонгации с учетом всех требований"""
#
#     # Запускаем существующий анализ (1-й коэффициент)
#     first_coeff_results = calculate_prolongation_metrics_improved()
#
#     # Загружаем данные для дополнительных расчетов
#     prolongations_data = pd.read_csv('prolongations.csv')
#     financial_data = pd.read_csv('financial_data.csv')
#     financial_long_prepared = prepare_financial_data(financial_data)
#
#     # Расчет второго коэффициента пролонгации
#     print("\n" + "=" * 60)
#     print("🔄 РАСЧЕТ ВТОРОГО КОЭФФИЦИЕНТА ПРОЛОНГАЦИИ")
#     print("=" * 60)
#
#     second_coeff_results_list = []
#     analysis_months_2023 = [month for month in sorted(financial_long_prepared['month'].unique())
#                             if month.startswith('2023')]
#
#     for month in analysis_months_2023[:6]:  # Анализируем первые 6 месяцев 2023
#         try:
#             second_coeff_data = calculate_second_prolongation_coefficient(month, financial_long_prepared)
#             second_coeff_results_list.append(second_coeff_data)
#         except Exception as e:
#             print(f"Ошибка при расчете второго коэффициента для {month}: {e}")
#
#     # Расчет по менеджерам
#     manager_results_df = calculate_manager_prolongation_metrics(financial_long_prepared, prolongations_data)
#
#     # Создание комплексного отчета
#     create_comprehensive_report(first_coeff_results, second_coeff_results_list, manager_results_df,
#                                 financial_long_prepared)
#
#     return first_coeff_results, second_coeff_results_list, manager_results_df
#
#
# # ОБНОВЛЯЕМ ЗАПУСК
# if __name__ == "__main__":
#     first_coeff, second_coeff, manager_results = calculate_complete_prolongation_analysis()
#
#     print("\n🎉 ПОЛНЫЙ АНАЛИЗ ЗАВЕРШЕН!")
#     print("=" * 60)
#     print("Созданные файлы:")
#     print("  1. improved_prolongation_analysis.png - Графики анализа")
#     print("  2. improved_prolongation_report.xlsx - Базовый отчет")
#     print("  3. comprehensive_prolongation_report.xlsx - Полный отчет")
#     print("\n📊 ОСНОВНЫЕ РЕЗУЛЬТАТЫ:")
#     print(f"  • Проанализировано месяцев: {len(first_coeff)}")
#     print(f"  • Рассчитано вторых коэффициентов: {len(second_coeff)}")
#     print(f"  • Проанализировано менеджеров: {manager_results['manager'].nunique() if len(manager_results) > 0 else 0}")
#     print("\n📈 РЕКОМЕНДАЦИИ ДЛЯ РУКОВОДИТЕЛЯ:")
#     print("  • Используйте comprehensive_prolongation_report.xlsx для детального анализа")
#     print("  • Сравните эффективность менеджеров по коэффициентам пролонгации")
#     print("  • Проанализируйте причины различий в первом и втором коэффициентах")
#     print("  • Разработайте план улучшения на основе выявленных закономерностей")
