stylesheet = """
    QMainWindow {
        background-color: #ffffff
    }

    QPushButton {
        background-color: #909190;
        border: 1px solid #4a4d4b;
        color: black;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 3px;
    }
    
    #accordionHeaderButton {
        background-color: none;
        border: none;
        color: black;
        padding: 0px 0px;
        text-align: left;
        text-decoration: none;
        font-size: 20px;
        font-weight: bolder;
        margin: 2px 2px;
    }

    QGridLayout {
        background: #909190; 
    }
    
    QLineEdit {
        border: 1px solid #4a4d4b; 
        border-radius: 0;
        text-align: right;
    }

    QComboBox {
        border: 1px solid #4a4d4b; 
        border-radius: 0;
    }
    
    QTabWidget QScrollArea {
        border: none;
    }
    
    QTabWidget::pane {
        border: 1px solid #4a4d4b;
        top:-1px; 
        background: rgb(245, 245, 245);
    } 

    QTabBar::tab {
        background: #909190; 
        border: 1px solid #4a4d4b; 
        padding: 10px;
        border-right: 0px;
    } 

    QTabBar::tab:selected { 
        background: #efefef; 
        margin-bottom: -1px; 
    }
    
    QTabBar::tab:last {
        border-right: 1px solid #4a4d4b;
    }

   
"""
