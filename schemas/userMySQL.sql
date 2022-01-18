CREATE TABLE IF NOT EXISTS User (
    userId INT(11) PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(32) NOT NULL,
    email VARCHAR(64) NOT NULL,
    phone VARCHAR(16) NOT NULL,
    password VARCHAR(64) NOT NULL
);
INSERT INTO User 
VALUES (1, 'admin','smarteye012@gmail.com','+65 91111111','$2b$12$g89sNQTCUsqE.zqhxaIV2eRbP7inr4FEVbn5m30AvE3fl//sIZsEy');