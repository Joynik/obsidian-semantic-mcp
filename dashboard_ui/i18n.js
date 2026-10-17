// Each row follows: English, Russian, Ukrainian, French, Spanish, German,
// Portuguese, Simplified Chinese, Japanese.
const LANGUAGES = ['en', 'ru', 'uk', 'fr', 'es', 'de', 'pt', 'zh', 'ja'];
const TRANSLATIONS = {
  'Dashboard': ['Dashboard', 'Панель', 'Панель', 'Tableau de bord', 'Panel', 'Dashboard', 'Painel', '仪表板', 'ダッシュボード'],
  'Overview': ['Overview', 'Обзор', 'Огляд', 'Vue générale', 'Resumen', 'Übersicht', 'Visão geral', '概览', '概要'],
  'Search': ['Search', 'Поиск', 'Пошук', 'Recherche', 'Buscar', 'Suche', 'Buscar', '搜索', '検索'],
  'Activity': ['Activity', 'Активность', 'Активність', 'Activité', 'Actividad', 'Aktivität', 'Atividade', '活动', 'アクティビティ'],
  'Workspace': ['Workspace', 'Рабочее пространство', 'Робочий простір', 'Espace de travail', 'Espacio de trabajo', 'Arbeitsbereich', 'Área de trabalho', '工作区', 'ワークスペース'],
  'Language': ['Language', 'Язык', 'Мова', 'Langue', 'Idioma', 'Sprache', 'Idioma', '语言', '言語'],
  'Browser language': ['Browser language', 'Язык браузера', 'Мова браузера', 'Langue du navigateur', 'Idioma del navegador', 'Browsersprache', 'Idioma do navegador', '浏览器语言', 'ブラウザーの言語'],
  'Live updates': ['Live updates', 'Автообновление', 'Автооновлення', 'Actualisation auto', 'Actualización automática', 'Automatische Aktualisierung', 'Atualização automática', '自动更新', '自動更新'],
  'Light theme': ['Light theme', 'Светлая тема', 'Світла тема', 'Thème clair', 'Tema claro', 'Helles Design', 'Tema claro', '浅色主题', 'ライトテーマ'],
  'Dark theme': ['Dark theme', 'Тёмная тема', 'Темна тема', 'Thème sombre', 'Tema oscuro', 'Dunkles Design', 'Tema escuro', '深色主题', 'ダークテーマ'],
  'Switch to light theme': ['Switch to light theme', 'Включить светлую тему', 'Увімкнути світлу тему', 'Passer au thème clair', 'Cambiar a tema claro', 'Zum hellen Design wechseln', 'Mudar para tema claro', '切换到浅色主题', 'ライトテーマに切り替え'],
  'Switch to dark theme': ['Switch to dark theme', 'Включить тёмную тему', 'Увімкнути темну тему', 'Passer au thème sombre', 'Cambiar a tema oscuro', 'Zum dunklen Design wechseln', 'Mudar para tema escuro', '切换到深色主题', 'ダークテーマに切り替え'],
  'Dashboard sections': ['Dashboard sections', 'Разделы панели', 'Розділи панелі', 'Sections du tableau de bord', 'Secciones del panel', 'Dashboard-Bereiche', 'Seções do painel', '仪表板栏目', 'ダッシュボードのセクション'],
  'OSM dashboard overview': ['OSM dashboard overview', 'Обзор панели OSM', 'Огляд панелі OSM', 'Vue générale du tableau OSM', 'Resumen del panel OSM', 'OSM-Dashboard-Übersicht', 'Visão geral do painel OSM', 'OSM 仪表板概览', 'OSM ダッシュボード概要'],
  'Monitor indexing, service health, and search across your vaults.': ['Monitor indexing, service health, and search across your vaults.', 'Следите за индексом, сервисами и поиском во всех хранилищах.', 'Стежте за індексом, службами й пошуком у всіх сховищах.', 'Suivez l’index, les services et la recherche dans vos coffres.', 'Supervisa el índice, los servicios y la búsqueda en tus bóvedas.', 'Index, Dienste und Suche in deinen Vaults überwachen.', 'Monitore o índice, os serviços e a busca nos seus cofres.', '监控所有笔记库的索引、服务状态和搜索。', '保管庫の索引、サービス、検索を監視します。'],
  'Services': ['Services', 'Сервисы', 'Служби', 'Services', 'Servicios', 'Dienste', 'Serviços', '服务', 'サービス'],
  'Local runtime status': ['Local runtime status', 'Состояние локальных служб', 'Стан локальних служб', 'État des services locaux', 'Estado de los servicios locales', 'Status der lokalen Dienste', 'Estado dos serviços locais', '本地服务状态', 'ローカルサービスの状態'],
  'Start': ['Start', 'Запустить', 'Запустити', 'Démarrer', 'Iniciar', 'Starten', 'Iniciar', '启动', '起動'],
  'Model — NOT LOADED': ['Model — NOT LOADED', 'Модель — НЕ ЗАГРУЖЕНА', 'Модель — НЕ ЗАВАНТАЖЕНА', 'Modèle — NON CHARGÉ', 'Modelo — SIN CARGAR', 'Modell — NICHT GELADEN', 'Modelo — NÃO CARREGADO', '模型 — 未加载', 'モデル — 未読込'],
  'DOWN': ['DOWN', 'НЕДОСТУПЕН', 'НЕДОСТУПНА', 'INDISPONIBLE', 'NO DISPONIBLE', 'NICHT ERREICHBAR', 'INDISPONÍVEL', '不可用', '停止中'],
  'INDEX': ['INDEX', 'ИНДЕКС', 'ІНДЕКС', 'INDEX', 'ÍNDICE', 'INDEX', 'ÍNDICE', '索引', 'インデックス'],
  'ANALYTICS': ['ANALYTICS', 'АНАЛИТИКА', 'АНАЛІТИКА', 'ANALYSE', 'ANÁLISIS', 'ANALYSE', 'ANÁLISE', '分析', '分析'],
  'EXPLORE': ['EXPLORE', 'ПОИСК', 'ПОШУК', 'EXPLORER', 'EXPLORAR', 'ERKUNDEN', 'EXPLORAR', '探索', '探索'],
  'Index health': ['Index health', 'Состояние индекса', 'Стан індексу', 'État de l’index', 'Estado del índice', 'Indexstatus', 'Estado do índice', '索引状态', 'インデックスの状態'],
  'Re-index': ['Re-index', 'Доиндексировать', 'Доіндексувати', 'Réindexer', 'Reindexar', 'Neu indexieren', 'Reindexar', '重新索引', '再インデックス'],
  'Safe Rebuild': ['Safe Rebuild', 'Безопасная перестройка', 'Безпечна перебудова', 'Reconstruction sûre', 'Reconstrucción segura', 'Sicherer Neuaufbau', 'Reconstrução segura', '安全重建', '安全に再構築'],
  'Indexing in progress — stats will update on completion': ['Indexing in progress — stats will update on completion', 'Идёт индексация — показатели обновятся после завершения', 'Триває індексація — показники оновляться після завершення', 'Indexation en cours — les chiffres suivront', 'Indexando — los datos se actualizarán al terminar', 'Indexierung läuft — Daten werden danach aktualisiert', 'Indexação em andamento — dados serão atualizados ao concluir', '正在索引 — 完成后更新统计', '索引中 — 完了後に統計を更新します'],
  'Indexed Notes': ['Indexed Notes', 'Заметок в индексе', 'Нотаток в індексі', 'Notes indexées', 'Notas indexadas', 'Indexierte Notizen', 'Notas indexadas', '已索引笔记', '索引済みノート'],
  'Vault Files': ['Vault Files', 'Файлы хранилищ', 'Файли сховищ', 'Fichiers des coffres', 'Archivos de bóvedas', 'Vault-Dateien', 'Arquivos dos cofres', '笔记库文件', '保管庫のファイル'],
  'Unindexed': ['Unindexed', 'Без индекса', 'Не індексовано', 'Non indexés', 'Sin indexar', 'Nicht indexiert', 'Não indexados', '未索引', '未索引'],
  'Orphaned Embeddings': ['Orphaned Embeddings', 'Осиротевшие векторы', 'Осиротілі вектори', 'Vecteurs orphelins', 'Vectores huérfanos', 'Verwaiste Vektoren', 'Vetores órfãos', '孤立向量', '孤立ベクトル'],
  'DB Size': ['DB Size', 'Размер БД', 'Розмір БД', 'Taille de la BD', 'Tamaño de la BD', 'DB-Größe', 'Tamanho do BD', '数据库大小', 'DB サイズ'],
  'Last Indexed': ['Last Indexed', 'Последняя индексация', 'Остання індексація', 'Dernière indexation', 'Última indexación', 'Letzte Indexierung', 'Última indexação', '上次索引', '最終索引'],
  'files not yet embedded': ['files not yet embedded', 'файлы без векторов', 'файли без векторів', 'fichiers sans vecteurs', 'archivos sin vectores', 'Dateien ohne Vektoren', 'arquivos sem vetores', '尚未生成向量的文件', 'ベクトル未生成のファイル'],
  'in DB but not on disk': ['in DB but not on disk', 'в БД, но не на диске', 'у БД, але не на диску', 'en BD, absents du disque', 'en BD, no en disco', 'in DB, nicht auf Datenträger', 'no BD, não no disco', '在数据库中但不在磁盘上', 'DB 内にありディスク上にない'],
  'Search activity': ['Search activity', 'Активность поиска', 'Активність пошуку', 'Activité de recherche', 'Actividad de búsqueda', 'Suchaktivität', 'Atividade de busca', '搜索活动', '検索アクティビティ'],
  'Rolling 24-hour window · UTC': ['Rolling 24-hour window · UTC', 'Последние 24 часа · UTC', 'Останні 24 години · UTC', 'Dernières 24 h · UTC', 'Últimas 24 h · UTC', 'Letzte 24 Std. · UTC', 'Últimas 24 h · UTC', '最近 24 小时 · UTC', '直近 24 時間 · UTC'],
  'Searches': ['Searches', 'Поисковые запросы', 'Пошукові запити', 'Recherches', 'Búsquedas', 'Suchanfragen', 'Buscas', '搜索次数', '検索回数'],
  'Current UTC hour and previous 23': ['Current UTC hour and previous 23', 'Текущий час UTC и 23 предыдущих', 'Поточна година UTC і 23 попередні', 'Heure UTC actuelle et 23 précédentes', 'Hora UTC actual y 23 anteriores', 'Aktuelle UTC-Stunde und 23 davor', 'Hora UTC atual e 23 anteriores', '当前 UTC 小时及之前 23 小时', '現在の UTC 時間と過去 23 時間'],
  'Search duration': ['Search duration', 'Время поиска', 'Тривалість пошуку', 'Durée de recherche', 'Duración de búsqueda', 'Suchdauer', 'Duração da busca', '搜索耗时', '検索時間'],
  'Effective backend': ['Effective backend', 'Используемый движок', 'Використаний рушій', 'Moteur utilisé', 'Motor utilizado', 'Verwendetes Backend', 'Mecanismo usado', '实际使用的后端', '使用したバックエンド'],
  'Search your notes': ['Search your notes', 'Поиск по заметкам', 'Пошук у нотатках', 'Chercher dans les notes', 'Buscar en tus notas', 'Notizen durchsuchen', 'Buscar nas notas', '搜索笔记', 'ノートを検索'],
  'Test search and recent indexing': ['Test search and recent indexing', 'Проверка поиска и недавняя индексация', 'Перевірка пошуку й недавня індексація', 'Test de recherche et indexations récentes', 'Prueba la búsqueda y revisa las indexaciones', 'Suche testen und letzte Indexierung prüfen', 'Teste a busca e veja indexações recentes', '测试搜索并查看最近索引', '検索テストと最近の索引'],
  'Test search': ['Test search', 'Проверить поиск', 'Перевірити пошук', 'Tester la recherche', 'Probar búsqueda', 'Suche testen', 'Testar busca', '测试搜索', '検索をテスト'],
  'Run a query against your indexed notes.': ['Run a query against your indexed notes.', 'Найдите данные в проиндексированных заметках.', 'Знайдіть дані в проіндексованих нотатках.', 'Lancez une recherche dans vos notes indexées.', 'Busca en tus notas indexadas.', 'Indexierte Notizen durchsuchen.', 'Busque nas notas indexadas.', '在已索引的笔记中搜索。', '索引済みノートを検索します。'],
  'Query': ['Query', 'Запрос', 'Запит', 'Requête', 'Consulta', 'Suchanfrage', 'Consulta', '查询', 'クエリ'],
  'What are you looking for?': ['What are you looking for?', 'Что вы ищете?', 'Що ви шукаєте?', 'Que cherchez-vous ?', '¿Qué buscas?', 'Wonach suchst du?', 'O que você procura?', '您要查找什么？', '何を探していますか？'],
  'Mode': ['Mode', 'Режим', 'Режим', 'Mode', 'Modo', 'Modus', 'Modo', '模式', 'モード'],
  'Hybrid': ['Hybrid', 'Гибридный', 'Гібридний', 'Hybride', 'Híbrido', 'Hybrid', 'Híbrido', '混合', 'ハイブリッド'],
  'Semantic': ['Semantic', 'Смысловой', 'Семантичний', 'Sémantique', 'Semántico', 'Semantisch', 'Semântico', '语义', '意味検索'],
  'Keyword': ['Keyword', 'По словам', 'За словами', 'Mots-clés', 'Palabras clave', 'Stichwort', 'Palavras-chave', '关键词', 'キーワード'],
  'Vault': ['Vault', 'Хранилище', 'Сховище', 'Coffre', 'Bóveda', 'Vault', 'Cofre', '笔记库', '保管庫'],
  'All vaults': ['All vaults', 'Все хранилища', 'Усі сховища', 'Tous les coffres', 'Todas las bóvedas', 'Alle Vaults', 'Todos os cofres', '所有笔记库', 'すべての保管庫'],
  'Limit': ['Limit', 'Количество', 'Кількість', 'Limite', 'Límite', 'Limit', 'Limite', '数量上限', '上限'],
  'Min similarity': ['Min similarity', 'Мин. сходство', 'Мін. схожість', 'Similarité min.', 'Similitud mín.', 'Min. Ähnlichkeit', 'Similaridade mín.', '最低相似度', '最小類似度'],
  'Search ↗': ['Search ↗', 'Искать ↗', 'Шукати ↗', 'Chercher ↗', 'Buscar ↗', 'Suchen ↗', 'Buscar ↗', '搜索 ↗', '検索 ↗'],
  'Recently indexed': ['Recently indexed', 'Недавно в индексе', 'Нещодавно в індексі', 'Indexés récemment', 'Indexadas recientemente', 'Zuletzt indexiert', 'Indexadas recentemente', '最近索引', '最近索引したノート'],
  'Toggle recent notes': ['Toggle recent notes', 'Показать или скрыть недавние заметки', 'Показати або сховати недавні нотатки', 'Afficher ou masquer les notes récentes', 'Mostrar u ocultar notas recientes', 'Letzte Notizen ein- oder ausblenden', 'Mostrar ou ocultar notas recentes', '显示或隐藏最近的笔记', '最近のノートを表示・非表示'],
  'Loading...': ['Loading...', 'Загрузка...', 'Завантаження...', 'Chargement...', 'Cargando...', 'Laden...', 'Carregando...', '加载中...', '読み込み中...'],
  'Fetching...': ['Fetching...', 'Получение данных...', 'Отримання даних...', 'Récupération...', 'Obteniendo datos...', 'Daten werden geladen...', 'Buscando dados...', '正在获取数据...', 'データ取得中...'],
  'No notes indexed yet': ['No notes indexed yet', 'Заметок в индексе пока нет', 'Нотаток в індексі ще немає', 'Aucune note indexée', 'Aún no hay notas indexadas', 'Noch keine Notizen indexiert', 'Nenhuma nota indexada', '尚无已索引笔记', '索引済みノートはまだありません'],
  'No results found.': ['No results found.', 'Ничего не найдено.', 'Нічого не знайдено.', 'Aucun résultat.', 'No se encontraron resultados.', 'Keine Ergebnisse.', 'Nenhum resultado.', '未找到结果。', '結果が見つかりません。'],
  'Results': ['Results', 'Результаты', 'Результати', 'Résultats', 'Resultados', 'Ergebnisse', 'Resultados', '结果', '結果'],
  'Errors': ['Errors', 'Ошибки', 'Помилки', 'Erreurs', 'Errores', 'Fehler', 'Erros', '错误', 'エラー'],
  'Unavailable': ['Unavailable', 'Недоступно', 'Недоступно', 'Indisponible', 'No disponible', 'Nicht verfügbar', 'Indisponível', '不可用', '利用不可'],
  'coverage': ['coverage', 'охват', 'охоплення', 'couverture', 'cobertura', 'Abdeckung', 'cobertura', '覆盖率', 'カバー率'],
  '.md files in vault': ['.md files in vault', 'файлы .md в хранилище', 'файли .md у сховищі', 'fichiers .md dans les coffres', 'archivos .md en las bóvedas', '.md-Dateien in Vaults', 'arquivos .md nos cofres', '笔记库中的 .md 文件', '保管庫内の .md ファイル'],
  'Last refresh': ['Last refresh', 'Обновлено', 'Оновлено', 'Actualisé', 'Actualizado', 'Aktualisiert', 'Atualizado', '上次刷新', '最終更新'],
  'Live updates every 30s': ['Live updates every 30s', 'Обновление каждые 30 с', 'Оновлення кожні 30 с', 'Actualisation toutes les 30 s', 'Actualización cada 30 s', 'Aktualisierung alle 30 s', 'Atualização a cada 30 s', '每 30 秒自动更新', '30 秒ごとに自動更新'],
  'Live updates paused': ['Live updates paused', 'Автообновление приостановлено', 'Автооновлення призупинено', 'Actualisation en pause', 'Actualización en pausa', 'Aktualisierung pausiert', 'Atualização pausada', '自动更新已暂停', '自動更新を一時停止'],
  'Service unreachable': ['Service unreachable', 'Сервис недоступен', 'Служба недоступна', 'Service inaccessible', 'Servicio inaccesible', 'Dienst nicht erreichbar', 'Serviço inacessível', '服务无法访问', 'サービスに接続できません'],
  'timed out': ['timed out', 'время ожидания истекло', 'час очікування вичерпано', 'délai dépassé', 'tiempo agotado', 'Zeitüberschreitung', 'tempo esgotado', '请求超时', 'タイムアウト'],
  'Dashboard disconnected': ['Dashboard disconnected', 'Панель отключена', 'Панель відключена', 'Tableau de bord déconnecté', 'Panel desconectado', 'Dashboard getrennt', 'Painel desconectado', '仪表板已断开连接', 'ダッシュボードが切断されました'],
  'Dashboard disconnected. Run': ['Dashboard disconnected. Run', 'Панель отключена. Запустите', 'Панель відключена. Запустіть', 'Tableau de bord déconnecté. Lancez', 'Panel desconectado. Ejecuta', 'Dashboard getrennt. Starte', 'Painel desconectado. Execute', '仪表板已断开连接。运行', 'ダッシュボードが切断されました。実行:'],
  'to start it again.': ['to start it again.', 'для повторного запуска.', 'для повторного запуску.', 'pour le relancer.', 'para iniciarlo de nuevo.', 'zum erneuten Start.', 'para iniciá-lo novamente.', '以重新启动。', 'で再起動します。'],
  'Build a new index, then switch to it? The previous index will be archived.': ['Build a new index, then switch to it? The previous index will be archived.', 'Создать новый индекс и переключиться на него? Предыдущий индекс будет сохранён в архиве.', 'Створити новий індекс і перейти на нього? Попередній індекс буде збережено в архіві.', 'Créer un nouvel index et l’activer ? L’ancien sera archivé.', '¿Crear y activar un índice nuevo? El anterior se archivará.', 'Neuen Index erstellen und aktivieren? Der bisherige wird archiviert.', 'Criar e ativar um novo índice? O anterior será arquivado.', '建立并启用新索引？旧索引将归档。', '新しいインデックスを作成して切り替えますか？以前のインデックスは保管されます。'],
  'Starting…': ['Starting…', 'Запуск…', 'Запуск…', 'Démarrage…', 'Iniciando…', 'Startet…', 'Iniciando…', '正在启动…', '起動中…'],
  'Running…': ['Running…', 'Выполняется…', 'Виконується…', 'En cours…', 'En curso…', 'Läuft…', 'Em execução…', '正在运行…', '実行中…'],
  'Started': ['Started', 'Запущен', 'Запущено', 'Démarré', 'Iniciado', 'Gestartet', 'Iniciado', '已启动', '起動済み'],
  'Failed': ['Failed', 'Не удалось', 'Не вдалося', 'Échec', 'Error', 'Fehlgeschlagen', 'Falha', '失败', '失敗'],
  'Error': ['Error', 'Ошибка', 'Помилка', 'Erreur', 'Error', 'Fehler', 'Erro', '错误', 'エラー'],
};

const LANGUAGE_KEY = 'osm-dashboard-language';
let currentLanguage = 'en';
const staticText = [];

function t(key) {
  const values = TRANSLATIONS[key];
  return values ? values[LANGUAGES.indexOf(currentLanguage)] : key;
}

function captureStaticText() {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const value = node.textContent || '';
    const key = value.trim();
    if (TRANSLATIONS[key]) {
      const start = value.indexOf(key);
      staticText.push([node, value.slice(0, start), key, value.slice(start + key.length)]);
    }
  }
}

function browserLanguage() {
  const preferences = [...(navigator.languages || []), navigator.language];
  for (const preference of preferences) {
    const language = preference.toLowerCase().split('-')[0];
    if (LANGUAGES.includes(language)) return language;
  }
  return 'en';
}

function applyLanguage(language) {
  const choice = LANGUAGES.includes(language) ? language : 'auto';
  currentLanguage = choice === 'auto' ? browserLanguage() : choice;
  document.documentElement.lang = currentLanguage === 'zh' ? 'zh-CN' : currentLanguage;
  document.getElementById('language-select').value = choice;
  staticText.forEach(([node, before, key, after]) => { node.textContent = before + t(key) + after; });
  document.title = 'OSM — ' + t('Dashboard');
  document.querySelector('.brand').setAttribute('aria-label', t('OSM dashboard overview'));
  document.querySelector('.primary-nav').setAttribute('aria-label', t('Dashboard sections'));
  document.getElementById('language-select').setAttribute('aria-label', t('Language'));
  document.getElementById('live-refresh').setAttribute('aria-label', t('Live updates'));
  document.getElementById('search-query').placeholder = t('What are you looking for?');
  document.getElementById('recent-toggle').setAttribute('aria-label', t('Toggle recent notes'));
}

function savedLanguage() {
  try {
    const saved = localStorage.getItem(LANGUAGE_KEY);
    if (LANGUAGES.includes(saved)) return saved;
  } catch (e) { /* Storage may be unavailable. */ }
  return 'auto';
}
