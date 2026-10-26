import csv

import pytest

from markup_helper.merge_reviewed import MergeError, main, merge

TRAINING = (
    'Filename;Temperature;Burn;Heating;HotWater;InternalPump;GasValve;SourceDir\r\n'
    'a.jpg;53;true;true;false;true;true;source_data/old\r\n'
    'b.jpg;47;false;false;false;false;false;source_data/old;\r\n'
    'c.jpg;;;;;;source_data/old;\r\n'
    'd.jpg;50;true;true;true;true;true;v'
)

REVIEWED_HEADER = ('Filename;Temperature;Burn;Heating;HotWater;InternalPump;GasValve;'
                   'SourceDir;ScreenState;too_dark;confirmed;retrainNeeded\n')
REVIEWED = REVIEWED_HEADER + (
    'n1.jpg;44;true;true;false;true;true;new;ok;false;true;\n'
    'n2.jpg;30;false;true;false;true;false;new;ok;false;false;Temperature:31>30\n'
    'n3.jpg;;;;;;;new;unreadable;true;true;\n'
    'n3.jpg;;;;;;;new;unreadable;true;true;\n'
    'n4.jpg;45;true;true;true;true;true;new;;;;\n'
)


@pytest.fixture
def root(tmp_path):
    for d, names in {'old': 'abcd', 'new': ['n1', 'n2', 'n3', 'n4']}.items():
        (tmp_path / 'source_data' / d).mkdir(parents=True)
        for n in names:
            (tmp_path / 'source_data' / d / f'{n}.jpg').write_bytes(b'')
    (tmp_path / 'training.csv').write_bytes(TRAINING.encode())
    (tmp_path / 'reviewed.csv').write_text(REVIEWED, newline='')
    return tmp_path


def run(root, **kw):
    kw.setdefault('source_dir', 'source_data/new')
    return merge(root / 'reviewed.csv', root / 'training.csv', root=root, **kw)


def read(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.reader(f, delimiter=';'))


def test_merge_appends_reviewed_rows(root):
    report = run(root, fix=True)
    rows = read(root / 'training.csv')

    assert rows[0][-1] == 'ScreenState' and len(rows[0]) == 9
    assert all(len(r) == 9 for r in rows)
    assert [r[0] for r in rows[1:]] == ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg',
                                        'n1.jpg', 'n2.jpg', 'n3.jpg']
    assert rows[6] == ['n2.jpg', '30', 'false', 'true', 'false', 'true', 'false',
                       'source_data/new', 'ok']
    assert rows[7][7:] == ['source_data/new', 'unreadable']
    assert report.added == {'ok': 2, 'unreadable': 1}
    assert report.corrected == 1
    assert report.skipped_pending == 1
    assert report.skipped_duplicate == 1
    assert report.header_upgraded


def test_training_rows_are_normalized(root):
    report = run(root, fix=True)
    rows = read(root / 'training.csv')

    assert rows[3] == ['c.jpg', '', '', '', '', '', '', 'source_data/old', '']
    assert rows[4][7] == 'source_data/old'
    assert report.realigned == [(4, 'c.jpg')]
    assert report.fixed_source_dir == [(5, 'v', 'source_data/old')]


def test_without_fix_bad_source_dir_is_only_reported(root):
    report = run(root)
    assert read(root / 'training.csv')[4][7] == 'v'
    assert report.unresolved == [(5, 'v/d.jpg')]


def test_line_endings_kept_and_backup_written(root):
    run(root)
    data = (root / 'training.csv').read_bytes()
    assert data.count(b'\r\n') == 8 and b'\n' not in data.replace(b'\r\n', b'')
    backups = list(root.glob('training.csv.bak-*'))
    assert len(backups) == 1 and backups[0].read_bytes() == TRAINING.encode()


def test_dry_run_writes_nothing(root):
    report = run(root, dry_run=True)
    assert sum(report.added.values()) == 3
    assert (root / 'training.csv').read_bytes() == TRAINING.encode()
    assert not list(root.glob('training.csv.*'))


def test_missing_image_aborts(root):
    (root / 'source_data' / 'new' / 'n2.jpg').unlink()
    with pytest.raises(MergeError, match='n2.jpg'):
        run(root)
    assert (root / 'training.csv').read_bytes() == TRAINING.encode()


def test_unresolved_source_dir_without_override_aborts(root):
    with pytest.raises(MergeError, match='images not found'):
        run(root, source_dir=None)


def test_conflicting_row_aborts(root):
    (root / 'reviewed.csv').write_text(
        REVIEWED_HEADER + 'a.jpg;54;true;true;false;true;true;x;ok;false;false;'
                          'Temperature:53>54\n', newline='')
    with pytest.raises(MergeError, match='different values'):
        run(root, source_dir='source_data/old')
    assert (root / 'training.csv').read_bytes() == TRAINING.encode()


def test_identical_existing_row_is_skipped(root):
    (root / 'reviewed.csv').write_text(
        REVIEWED_HEADER + 'a.jpg;53;true;true;false;true;true;x;;false;true;\n', newline='')
    report = run(root, source_dir='source_data/old')
    assert report.skipped_duplicate == 1 and not report.added


def test_cli_reports_error(root, capsys):
    (root / 'source_data' / 'new' / 'n1.jpg').unlink()
    code = main(['--reviewed', str(root / 'reviewed.csv'),
                 '--training', str(root / 'training.csv'), '--source-dir', 'x'])
    assert code == 1
    assert 'Nothing was written' in capsys.readouterr().err
