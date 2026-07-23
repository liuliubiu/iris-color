-- 本地 MySQL 初始化（在 mysql 客户端执行）
CREATE DATABASE IF NOT EXISTS iris_experiment
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE iris_experiment;

-- 表结构由 iris-vision 首次启动时自动创建；也可手动执行：
CREATE TABLE IF NOT EXISTS experiment_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    group_name VARCHAR(32) NOT NULL COMMENT '实验大组 G{YYYYMMDD}-{NNN}',
    subgroup_name VARCHAR(16) NULL COMMENT '实验小组 S{NN}，可空',
    experiment_date DATE NOT NULL,
    operator VARCHAR(64) NOT NULL,
    camera_device VARCHAR(128) NULL,
    light_device VARCHAR(128) NULL,
    illuminance INT NULL COMMENT '环境照度 lx',
    color VARCHAR(16) NULL COMMENT '9种标准色',
    grade_before VARCHAR(16) NULL,
    lstar_before DOUBLE NULL,
    grade_after VARCHAR(16) NULL,
    lstar_after DOUBLE NULL,
    notes TEXT NULL,
    image_rel VARCHAR(512) NULL COMMENT 'img/ 相对路径',
    debug_run_id VARCHAR(32) NULL COMMENT 'debug_output run_id',
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    INDEX idx_exp_group (group_name),
    INDEX idx_exp_date (experiment_date),
    INDEX idx_exp_operator (operator)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
