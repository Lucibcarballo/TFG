function doPost(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

  var atributos = ['Brillantez', 'Proyección', 'Sustain', 'Cuerpo', 'Claridad', 'Equilibrio'];
  var datos = [];

  // 1. Fecha y Hora
  datos.push(new Date());

  // 2. Nombre del Evaluador
  var nombre = e.parameter['Nombre del Evaluador'];
  datos.push(nombre);

  // 3. Recorremos cada atributo y cada PUESTO (del 1 al 5)
  // En el HTML el name es: "Atributo - X Puesto"
  for (var i = 0; i < atributos.length; i++) {
    for (var j = 1; j <= 5; j++) {
      var key = atributos[i] + ' - ' + j + ' Puesto'; // Cambio clave aquí
      var valor = e.parameter[key] || '';
      datos.push(valor);
    }
  }

  // 4. Comentarios finales
  var comentario = e.parameter['Comentarios'];
  datos.push(comentario);

  // Guardar fila completa
  sheet.appendRow(datos);

  return ContentService.createTextOutput("ok");
}