function doPost(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

  var atributos = ['Brillantez', 'Proyección', 'Sustain', 'Cuerpo', 'Claridad', 'Equilibrio'];
  var datos = [];

  datos.push(new Date());

  var nombre = e.parameter['Nombre del Evaluador']
  datos.push(nombre)

  // Recorremos cada atributo y cada audio
  for (var i = 0; i < atributos.length; i++) {
    for (var j = 1; j <= 10; j++) {
      var key = atributos[i] + ' - Audio ' + j;
      var valor = e.parameter[key] || ''; // por si viene vacío
      datos.push(valor);
    }
  }

  var comentario = e.parameter['Comentarios']
  datos.push(comentario)


  // Guardar fila completa
  sheet.appendRow(datos);

  return ContentService.createTextOutput("ok");
}